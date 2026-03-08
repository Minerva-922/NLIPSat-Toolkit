import os
import subprocess
import tempfile
import time
from math import gcd
from functools import reduce
from pysat.formula import WCNF, IDPool
from pysat.examples.rc2 import RC2
from encoders.encoder_driver import encode_objective, add_domain_constraints, encode_constraints
from .config import EncodingConfig
from tools.preprocessing import preprocess_problem


class EncodingTimeoutError(RuntimeError):
    pass


def _preprocess_wcnf(wcnf):
    if not wcnf.hard:
        return

    orig_hard = len(wcnf.hard)

    # Unit propagation on hard clauses
    units = set()
    for clause in wcnf.hard:
        if len(clause) == 1:
            units.add(clause[0])

    if units:
        new_hard = []
        for clause in wcnf.hard:
            if any(lit in units for lit in clause):
                if len(clause) == 1:
                    new_hard.append(clause)
                continue
            new_clause = [lit for lit in clause if -lit not in units]
            if new_clause:
                new_hard.append(new_clause)
        wcnf.hard = new_hard

    # Subsumption on hard clauses: remove clauses subsumed by shorter ones.
    # Only process if clause count is manageable to avoid quadratic blowup.
    if len(wcnf.hard) < 500_000:
        clause_sets = [(frozenset(c), i) for i, c in enumerate(wcnf.hard)]
        clause_sets.sort(key=lambda x: len(x[0]))
        subsumed = set()
        short_clauses = []
        for cs, idx in clause_sets:
            if idx in subsumed:
                continue
            for sc, _ in short_clauses:
                if len(sc) < len(cs) and sc.issubset(cs):
                    subsumed.add(idx)
                    break
            if idx not in subsumed and len(cs) <= 3:
                short_clauses.append((cs, idx))

        if subsumed:
            wcnf.hard = [c for i, c in enumerate(wcnf.hard) if i not in subsumed]

    removed = orig_hard - len(wcnf.hard)
    if removed > 0:
        print(f"  [wcnf-preprocess] removed {removed} hard clauses "
              f"(units: {len(units)}, subsumption applied)")


def _add_lp_relaxation_bound(problem, wcnf, vpool, encoding, weight_gcd):
    variables = problem.get('variables', {}) or {}
    for info in variables.values():
        if int(info.get('lb', 0)) != 0 or int(info.get('ub', 0)) != 1:
            return

    terms = problem.get('objective', {}).get('terms', [])
    if not terms:
        return

    # Compute LP relaxation lower bound: for each term, the LP min over [0,1]^n
    # with McCormick envelopes.  For uncorrelated terms (no LP coupling):
    #   linear c*x: min = min(0, c)
    #   quadratic c*x*y: LP min = min(0, c) (McCormick: w in [0,1], LP can set w=0 if c>0, w=1 if c<0)
    # This is a valid but loose bound. More importantly, we derive an upper bound
    # on the MaxSAT cost to add as a hard PB constraint.
    lp_lb = 0
    for t in terms:
        c_val = int(t.get('c', 0))
        if c_val < 0:
            lp_lb += c_val

    neg_offset = getattr(wcnf, '_neg_soft_total', 0)
    total_soft = sum(wcnf.wght)

    # Objective = (total_soft - cost - neg_offset) * weight_gcd + const
    # LP_lb <= objective  =>  cost <= total_soft - neg_offset - lp_lb / weight_gcd
    if weight_gcd > 0:
        max_cost = total_soft - neg_offset - (lp_lb // weight_gcd)
    else:
        return

    if max_cost >= total_soft or max_cost <= 0:
        return

    wcnf._lp_cost_bound = max_cost
    print(f"  [lp-bound] QBB LP relaxation bound: max_cost={max_cost} "
          f"(total={total_soft}, lp_lb={lp_lb})")


def _check_encoding_deadline(cfg, phase_name=""):
    deadline = getattr(cfg, 'encoding_deadline', 0.0)
    if deadline > 0 and time.time() > deadline:
        raise EncodingTimeoutError(
            f"Encoding deadline exceeded during {phase_name}")


def build_wcnf(problem, encoding, cfg, collect_timings=False):
    timings = {} if collect_timings else None

    # 0. Preprocessing
    t0 = time.time()
    preprocess_meta = None
    if getattr(cfg, 'enable_preprocess', False):
        processed_problem, meta = preprocess_problem(
            problem,
            denom_cap=10**6,
            scale_limit=getattr(cfg, 'preprocess_scale_limit', 10**6),
            enable_min_to_max=True,
            enable_integerize=getattr(cfg, 'preprocess_integerize', True)
        )
        problem = processed_problem
        preprocess_meta = meta
        print("  [preprocess] enabled: variable shift / normalization / integerization / objective normalization")
    if collect_timings:
        timings['preprocess'] = time.time() - t0
    _check_encoding_deadline(cfg, "preprocessing")

    vpool = IDPool()
    wcnf = WCNF()

    var_names = sorted(problem.get('variables', {}).keys())
    name2idx = {name: i + 1 for i, name in enumerate(var_names)}

    # Global AND-gate cache shared across constraints and objective (BIN encoding)
    if encoding == 'BIN':
        cfg._global_and_cache = {}

    # 1. Domain constraints
    t1 = time.time()
    add_domain_constraints(problem, encoding, name2idx, vpool, wcnf, cfg)
    if collect_timings:
        timings['domain_constraints'] = time.time() - t1
    _check_encoding_deadline(cfg, "domain constraints")

    # 2. Problem constraints
    t2 = time.time()
    wcnf_cons = encode_constraints(problem, encoding, name2idx, vpool, cfg)
    wcnf.extend(wcnf_cons.hard)
    if collect_timings:
        timings['problem_constraints'] = time.time() - t2
    _check_encoding_deadline(cfg, "problem constraints")

    # 3. Objective function
    t3 = time.time()
    wcnf_obj = encode_objective(problem, encoding, name2idx, vpool, cfg)
    wcnf.extend(wcnf_obj.hard)
    wcnf.extend(wcnf_obj.soft, wcnf_obj.wght)
    _obj_elapsed = time.time() - t3
    if collect_timings:
        timings['objective_encoding'] = _obj_elapsed
    _check_encoding_deadline(cfg, "objective encoding")

    if encoding == 'BIN' and hasattr(cfg, '_global_and_cache'):
        cache_size = len(cfg._global_and_cache)
        if cache_size > 0:
            print(f"  [global-cache] shared AND-gate cache: {cache_size} gates")
        del cfg._global_and_cache

    # Constant terms in objective (terms with empty vars)
    obj_constant = 0
    for term in problem.get('objective', {}).get('terms', []):
        if not term.get('vars'):
            obj_constant += int(term.get('c', 0))
    wcnf._objective_constant = obj_constant

    # Record negative-weight offset
    if hasattr(wcnf_obj, '_neg_soft_total'):
        wcnf._neg_soft_total = getattr(wcnf_obj, '_neg_soft_total', 0)

    # Soft weight GCD normalization (Direction 1/5: core-friendly)
    weight_gcd = 1
    if getattr(cfg, 'weight_gcd_normalize', True) and wcnf.wght:
        weight_gcd = reduce(gcd, wcnf.wght)
        if weight_gcd > 1:
            wcnf.wght = [w // weight_gcd for w in wcnf.wght]
            if hasattr(wcnf, '_neg_soft_total'):
                wcnf._neg_soft_total = wcnf._neg_soft_total // weight_gcd
            print(f"  [weight-norm] GCD={weight_gcd}, weights reduced {weight_gcd}x")
    wcnf._weight_gcd = weight_gcd

    # For unconstrained binary quadratic problems (QBB), add an implied upper
    # bound constraint on the MaxSAT cost based on LP relaxation.
    constraints = problem.get('constraints', []) or []
    if not constraints and wcnf.wght:
        _add_lp_relaxation_bound(problem, wcnf, vpool, encoding, weight_gcd)

    # WCNF-level preprocessing: unit propagation and subsumption
    _preprocess_wcnf(wcnf)

    wcnf.topw = sum(wcnf.wght) + 1

    print(f"  Vars: {wcnf.nv}, Hard: {len(wcnf.hard)}, Soft: {len(wcnf.soft)}, Top: {wcnf.topw}")

    # Attach preprocessing metadata and processed problem
    wcnf._preprocess_meta = preprocess_meta
    wcnf._vpool = vpool
    wcnf._processed_problem = problem

    if collect_timings:
        timings['build_total'] = time.time() - t0
        return wcnf, name2idx, timings, vpool
    return wcnf, name2idx, vpool


def _greedy_to_phases(greedy_sol, problem, encoding, vpool):
    """Convert a greedy variable assignment to SAT phase hints."""
    import math
    if not greedy_sol:
        return []
    phases = []
    variables = problem.get('variables', {})
    var_names = sorted(variables.keys())
    name2idx = {name: i + 1 for i, name in enumerate(var_names)}
    for vn, val in greedy_sol.items():
        if vn not in name2idx:
            continue
        q = name2idx[vn]
        ub = int(variables[vn].get('ub', 0))
        val = max(0, min(int(val), ub))
        if encoding == 'OH':
            for r in range(ub + 1):
                vid = vpool.id(f'x_{q}@{r}')
                phases.append(vid if r == val else -vid)
        elif encoding == 'UNA':
            for r in range(ub + 1):
                vid = vpool.id(f'x_{q}@{r}')
                phases.append(vid if r <= val else -vid)
        elif encoding == 'BIN':
            nb = math.ceil(math.log2(ub + 1)) if ub > 0 else 1
            for r in range(nb):
                vid = vpool.id(f'x_{q}@{r}')
                phases.append(vid if (val >> r) & 1 else -vid)
    return phases


def solve_rc2(wcnf, phases=None):
    try:
        solver = RC2(wcnf, adapt=True, exhaust=True, trim=3)
        if phases:
            try:
                solver.oracle.set_phases(phases)
            except Exception:
                pass
        solution = solver.compute()

        if solution is None:
            return -1, []

        cost = solver.cost
        assignment = solution
        solver.delete()

        return cost, assignment

    except Exception as e:
        raise RuntimeError(f"RC2 failed: {e}")


def _compute_greedy_cost(wcnf, phases):
    """Compute MaxSAT cost of the greedy solution from phase hints.
    Counts the total weight of soft clauses unsatisfied by the assignment."""
    if not phases:
        return None
    assign_set = set(phases)
    cost = 0
    for clause, weight in zip(wcnf.soft, wcnf.wght):
        satisfied = any(lit in assign_set for lit in clause)
        if not satisfied:
            cost += weight
    return cost


def solve_external(wcnf, workdir, solver_path, timeout=0, greedy_cost=None):
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.wcnf',
        dir=workdir,
        delete=False
    ) as f:
        wcnf_path = f.name
        wcnf.to_file(wcnf_path)

    print(f"  WCNF file: {wcnf_path}")

    solver_name = os.path.basename(solver_path).lower()
    cmd = [solver_path]
    if timeout and timeout > 0:
        if solver_name == 'maxhs':
            cmd.append(f"-cpu-lim={int(timeout)}")
        elif solver_name in ('wmaxcdcl', 'openwbo'):
            cmd.append(f"-cpu-lim={int(timeout)}")
    if greedy_cost is not None and greedy_cost >= 0:
        print(f"  [warm-start] initial upper bound available: {greedy_cost} (not injected as CLI flag)")
    cmd.append(wcnf_path)

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=(int(timeout) + 30) if timeout and timeout > 0 else None,
    )

    # Parse the solver's status line (s ...) from stdout
    solver_status_line = ''
    for line in (proc.stdout or '').split('\n'):
        if line.startswith('s '):
            solver_status_line = line[2:].strip()

    # Accepted return codes:
    #   0  — normal exit (no conclusion / UNKNOWN)
    #   1  — timeout/interrupted for MAXHS and WMAXCDCL (cpu-lim reached)
    #   10 — SATISFIABLE (suboptimal feasible solution)
    #   20 — UNSATISFIABLE (hard clauses infeasible)
    #   30 — OPTIMUM FOUND
    if proc.returncode not in [0, 1, 10, 20, 30]:
        stdout_tail = "\n".join((proc.stdout or "").splitlines()[-8:])
        stderr_tail = "\n".join((proc.stderr or "").splitlines()[-8:])
        raise RuntimeError(
            f"External solver failed: {os.path.basename(solver_path)}, "
            f"rc={proc.returncode}\n"
            f"[stdout-tail]\n{stdout_tail}\n"
            f"[stderr-tail]\n{stderr_tail}"
        )

    # Map MaxSAT solver output to canonical status names used by the
    # analysis pipeline: OPTIMAL / FEASIBLE / UNSAT / TIMEOUT.
    if solver_status_line == 'OPTIMUM FOUND' or proc.returncode == 30:
        solver_status = 'OPTIMAL'
    elif solver_status_line == 'SATISFIABLE' or proc.returncode == 10:
        solver_status = 'FEASIBLE'
    elif solver_status_line == 'UNSATISFIABLE' or proc.returncode == 20:
        solver_status = 'UNSAT'
    else:
        solver_status = 'TIMEOUT'

    cost = -1
    assignment = []

    for line in proc.stdout.split('\n'):
        if line.startswith('o '):
            cost = int(line.split(' ')[1])
        if line.startswith('v '):
            assignment_str = line.split(' ')[1:]
            assignment = [int(v) for v in assignment_str if v and v != '0']

    # If solver timed out but found a feasible upper bound, upgrade to FEASIBLE
    if solver_status == 'TIMEOUT' and cost >= 0:
        solver_status = 'FEASIBLE'

    return cost, assignment, solver_status


def build_and_solve(problem, encoding, cfg, solver='RC2', collect_timings=True):
    t_start = time.time()

    # Build WCNF
    build_result = build_wcnf(problem, encoding, cfg, collect_timings=collect_timings)
    if collect_timings:
        wcnf, name2idx, build_timings, vpool = build_result
    else:
        wcnf, name2idx, vpool = build_result
        build_timings = {}

    # Compute phase hints from greedy solution (use preprocessed problem for
    # consistent variable names/bounds with the encoding)
    phases = None
    preprocess_meta = getattr(wcnf, '_preprocess_meta', None)
    processed_problem = getattr(wcnf, '_processed_problem', problem)
    if preprocess_meta is not None:
        greedy_sol = preprocess_meta.get('objective', {}).get('greedy_solution')
        if greedy_sol:
            phases = _greedy_to_phases(greedy_sol, processed_problem, encoding, vpool)

    greedy_cost = _compute_greedy_cost(wcnf, phases) if phases else None
    lp_bound = getattr(wcnf, '_lp_cost_bound', None)
    if lp_bound is not None:
        if greedy_cost is None or lp_bound < greedy_cost:
            greedy_cost = lp_bound

    # Solve
    t_solve = time.time()
    solver_upper = solver.upper()
    ext_solver_status = None
    if solver_upper == 'RC2':
        cost, assignment = solve_rc2(wcnf, phases=phases)
        ext_solver_status = 'OPTIMAL' if cost >= 0 else 'TIMEOUT'
    elif solver_upper in ('MAXHS', 'WMAXCDCL', 'OPENWBO'):
        path_map = {
            'MAXHS': cfg.maxhs_path,
            'WMAXCDCL': cfg.wmaxcdcl_path,
            'OPENWBO': cfg.openwbo_path,
        }
        cost, assignment, ext_solver_status = solve_external(
            wcnf,
            cfg.workdir,
            path_map[solver_upper],
            timeout=getattr(cfg, 'external_solver_timeout', 0),
            greedy_cost=greedy_cost
        )
    else:
        raise ValueError(f"Unsupported solver: {solver}")
    solve_time = time.time() - t_solve

    # Compute objective value from MaxSAT cost
    total_soft_weight = sum(wcnf.wght) if wcnf.wght else 0
    neg_offset = getattr(wcnf, '_neg_soft_total', 0)
    weight_gcd = getattr(wcnf, '_weight_gcd', 1)
    objective_value = ((total_soft_weight - cost) - neg_offset) * weight_gcd

    # Add constant terms from objective (not encoded)
    obj_constant = getattr(wcnf, '_objective_constant', 0)
    objective_value += obj_constant

    # Reverse preprocessing transformations to recover original objective value
    preprocess_meta = getattr(wcnf, '_preprocess_meta', None)
    if preprocess_meta is not None:
        const_shift = preprocess_meta.get('objective', {}).get('constant_shift', 0)
        sense_flip = preprocess_meta.get('objective', {}).get('sense_flip', False)
        scale_lcm = preprocess_meta.get('objective', {}).get('scale_lcm', 1)
        scale_gcd = preprocess_meta.get('objective', {}).get('scale_gcd', 1)

        if scale_lcm > 1 or scale_gcd > 1:
            from fractions import Fraction
            scale_frac = Fraction(scale_lcm, scale_gcd)
            scaled_const = int(Fraction(const_shift) * scale_frac)
        else:
            scaled_const = int(const_shift)
        objective_value += scaled_const

        if sense_flip:
            objective_value = -objective_value

        if scale_lcm > 1 or scale_gcd > 1:
            from fractions import Fraction
            inv_scale = Fraction(scale_gcd, scale_lcm)
            objective_value = float(Fraction(objective_value) * inv_scale)
            if objective_value == int(objective_value):
                objective_value = int(objective_value)

    total_time = time.time() - t_start

    use_decomp = (
        encoding == 'BIN' and
        cfg is not None and
        getattr(cfg, 'use_decomposition', False)
    )

    result = {
        'encoding': encoding,
        'use_decomposition': use_decomp,
        'solver': solver,
        'solver_status': ext_solver_status,
        'objective_value': objective_value,
        'maxsat_cost': cost,
        'total_soft_weight': total_soft_weight,
        'num_variables': wcnf.nv,
        'num_hard_clauses': len(wcnf.hard),
        'num_soft_clauses': len(wcnf.soft),
        'top_weight': wcnf.topw,
        'assignment': assignment,
        'name2idx': name2idx,
        'vpool': vpool
    }

    if collect_timings:
        result['timings'] = {
            **build_timings,
            'solve': solve_time,
            'total': total_time
        }

    if hasattr(wcnf, '_preprocess_meta') and wcnf._preprocess_meta is not None:
        result['preprocess_meta'] = wcnf._preprocess_meta

    return result