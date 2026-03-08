#!/usr/bin/env python3
"""Z3 baseline solver for QPLIB and SMT2 instances.

Pipeline for QPLIB:
  QPLIB file  →  qplib_parser  →  Python dict (variables, objective, constraints)
  →  Z3 Int variables + quadratic/linear constraints + objective  →  Z3 solve

Two-phase solving strategy:
  Phase 1 – Optimize(): fast for small / linear problems, can prove optimality.
  Phase 2 – Solver() + iterative tightening: more robust for nonlinear integer
            arithmetic where Optimize cannot determine bounds.
"""

import argparse
import json
import sys
import os
import time
import traceback
from fractions import Fraction
from pathlib import Path

CODES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "codes")
sys.path.insert(0, CODES_DIR)

try:
    from z3 import Int, IntVal, Optimize, Solver, SolverFor, sat, unsat, unknown, RatVal
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False


# ---------------------------------------------------------------------------
#  Parsing helpers
# ---------------------------------------------------------------------------

def load_instance(filepath, k=None):
    if filepath.endswith('.qplib'):
        from tools.qplib_parser import parse_qplib_file
        return parse_qplib_file(filepath)
    elif filepath.endswith('.smt2'):
        return None
    elif filepath.endswith('.cnf'):
        if k is not None and k >= 2:
            from tools.diversesat_parser import parse_diversesat
            return parse_diversesat(filepath, k=k)
        from tools.cnf_parser import parse_cnf_file
        return parse_cnf_file(filepath)
    else:
        with open(filepath) as f:
            return json.load(f)


# ---------------------------------------------------------------------------
#  Z3 expression building
# ---------------------------------------------------------------------------

def to_z3_coeff(c):
    """Convert a Python number to Z3-compatible int or RatVal."""
    f = Fraction(c).limit_denominator(10**15)
    return f.numerator if f.denominator == 1 else RatVal(f.numerator, f.denominator)


def build_expr(terms, z3_vars):
    """Build a Z3 arithmetic expression from a list of polynomial terms.

    Each term is {'c': coeff, 'vars': {'x1': power1, 'x2': power2, ...}}.
    """
    expr = 0
    for term in terms:
        c = to_z3_coeff(term.get('c', 1))
        tvars = term.get('vars', {})
        if not tvars:
            expr = expr + c
            continue
        monomial = c
        for vname, power in tvars.items():
            if vname in z3_vars:
                for _ in range(int(power)):
                    monomial = monomial * z3_vars[vname]
        expr = expr + monomial
    return expr


def eval_objective(model, obj_terms, z3_vars):
    """Evaluate objective at a Z3 model using exact Fraction arithmetic."""
    val = Fraction(0)
    for term in obj_terms:
        c = Fraction(term.get('c', 1)).limit_denominator(10**15)
        tvars = term.get('vars', {})
        if not tvars:
            val += c
            continue
        tv = c
        for vname, power in tvars.items():
            if vname in z3_vars:
                v = model[z3_vars[vname]]
                if v is not None:
                    tv *= Fraction(v.as_long()) ** int(power)
        val += tv
    r = float(val)
    return int(r) if r == int(r) else r


# ---------------------------------------------------------------------------
#  Z3 model construction helpers
# ---------------------------------------------------------------------------

def create_z3_vars(problem):
    """Create Z3 Int variables with domain bounds."""
    z3_vars = {}
    bounds = []
    for vname, info in problem.get('variables', {}).items():
        x = Int(vname)
        z3_vars[vname] = x
        bounds.append(x >= int(info.get('lb', 0)))
        bounds.append(x <= int(info.get('ub', 100)))
    return z3_vars, bounds


def add_problem_constraints(ctx, problem, z3_vars):
    """Add all parsed constraints (linear or quadratic) to a Z3 context."""
    for constr in problem.get('constraints', []):
        lhs = build_expr(constr.get('terms', []), z3_vars)
        rhs = to_z3_coeff(constr.get('rhs', 0))
        rel = constr.get('rel', constr.get('sense', '<='))
        if   rel == '<=':       ctx.add(lhs <= rhs)
        elif rel == '>=':       ctx.add(lhs >= rhs)
        elif rel in ('==', '='): ctx.add(lhs == rhs)
        elif rel == '<':        ctx.add(lhs < rhs)
        elif rel == '>':        ctx.add(lhs > rhs)


def _handle_is_finite(handle):
    """Check whether an Optimize handle has a concrete (finite) value."""
    s = str(handle.value())
    return s not in ('oo', '-1*oo', '+oo', '-oo', 'epsilon', '-epsilon')


# ---------------------------------------------------------------------------
#  Phase 1 – Optimize()
# ---------------------------------------------------------------------------

def _phase1_optimize(problem, z3_vars, bounds, obj_expr, obj_terms, is_max,
                     timeout):
    """Try Z3 Optimize; returns result dict or None if inconclusive."""
    opt = Optimize()
    opt.set("timeout", int(timeout * 1000))

    for b in bounds:
        opt.add(b)
    add_problem_constraints(opt, problem, z3_vars)

    handle = opt.maximize(obj_expr) if is_max else opt.minimize(obj_expr)
    result = opt.check()

    if result == sat and _handle_is_finite(handle):
        obj_val = eval_objective(opt.model(), obj_terms, z3_vars)
        return {'status': 'OPTIMAL', 'objective': obj_val, 'proven': True}

    if result == unsat:
        return {'status': 'UNSAT', 'objective': None, 'proven': True}

    best_val = None
    best_z3_obj = None
    if result == sat:
        m = opt.model()
        best_val = eval_objective(m, obj_terms, z3_vars)
        best_z3_obj = m.eval(obj_expr, model_completion=True)

    return {'status': None, 'best_val': best_val, 'best_z3_obj': best_z3_obj}


# ---------------------------------------------------------------------------
#  Phase 2 – Solver + iterative tightening
# ---------------------------------------------------------------------------

def _phase2_iterative(problem, z3_vars, bounds, obj_expr, obj_terms, is_max,
                      timeout_deadline, initial_val=None, initial_z3_obj=None):
    """Iteratively improve the objective with Z3 Solver.

    Returns (proven_optimal: bool, best_val, best_z3_obj).
    """
    solver = Solver()
    for b in bounds:
        solver.add(b)
    add_problem_constraints(solver, problem, z3_vars)

    best_val = initial_val
    best_z3_obj = initial_z3_obj
    proven = False

    while True:
        remaining_ms = int((timeout_deadline - time.perf_counter()) * 1000)
        if remaining_ms < 500:
            break

        solver.set("timeout", remaining_ms)
        solver.push()

        if best_z3_obj is not None:
            if is_max:
                solver.add(obj_expr > best_z3_obj)
            else:
                solver.add(obj_expr < best_z3_obj)

        r = solver.check()

        if r == sat:
            m = solver.model()
            new_val = eval_objective(m, obj_terms, z3_vars)
            new_z3_obj = m.eval(obj_expr, model_completion=True)
            improved = (best_val is None
                        or (is_max and new_val > best_val)
                        or (not is_max and new_val < best_val))
            if improved:
                best_val = new_val
                best_z3_obj = new_z3_obj
            solver.pop()
        elif r == unsat:
            solver.pop()
            if best_val is not None:
                proven = True
            break
        else:
            solver.pop()
            break

    return proven, best_val, best_z3_obj


# ---------------------------------------------------------------------------
#  Main QPLIB solver entry
# ---------------------------------------------------------------------------

def solve_qplib(problem, timeout):
    """Solve a parsed QPLIB instance with Z3 (two-phase strategy)."""
    wall_start = time.perf_counter()
    deadline = wall_start + timeout

    obj_terms = problem.get('objective', {}).get('terms', [])
    sense = problem.get('objective', {}).get('sense', 'min')
    is_max = (sense == 'max')

    z3_vars, bounds = create_z3_vars(problem)
    obj_expr = build_expr(obj_terms, z3_vars)
    if isinstance(obj_expr, int):
        obj_expr = IntVal(obj_expr)

    # ── Phase 1: Optimize() ──────────────────────────────────────
    # Cap Phase 1 time so Phase 2 always gets a fair share.
    phase1_budget = min(timeout * 0.4, 600)
    p1 = _phase1_optimize(problem, z3_vars, bounds, obj_expr, obj_terms,
                          is_max, phase1_budget)

    if p1.get('proven'):
        p1['time'] = time.perf_counter() - wall_start
        return p1

    best_val = p1.get('best_val')
    best_z3_obj = p1.get('best_z3_obj')

    # ── Phase 2: Solver + iterative tightening ───────────────────
    remaining = deadline - time.perf_counter()
    if remaining < 2:
        elapsed = time.perf_counter() - wall_start
        if best_val is not None:
            return {'status': 'FEASIBLE', 'objective': best_val,
                    'time': elapsed}
        return {'status': 'TIMEOUT' if elapsed >= timeout * 0.95 else 'UNKNOWN',
                'objective': None, 'time': elapsed}

    proven, best_val, _ = _phase2_iterative(
        problem, z3_vars, bounds, obj_expr, obj_terms, is_max,
        deadline, best_val, best_z3_obj)

    elapsed = time.perf_counter() - wall_start

    if proven:
        return {'status': 'OPTIMAL', 'objective': best_val, 'time': elapsed}
    if best_val is not None:
        return {'status': 'FEASIBLE', 'objective': best_val, 'time': elapsed}
    if elapsed >= timeout * 0.95:
        return {'status': 'TIMEOUT', 'objective': None, 'time': elapsed}
    return {'status': 'UNKNOWN', 'objective': None, 'time': elapsed}


# ---------------------------------------------------------------------------
#  SMT2 solver (satisfiability only) — multi-strategy portfolio
# ---------------------------------------------------------------------------

def _try_solver(solver, filepath, timeout_ms):
    """Run a single Z3 solver attempt. Returns z3 result enum."""
    solver.set("timeout", int(timeout_ms))
    solver.from_file(filepath)
    return solver.check()


def solve_smt2(filepath, timeout):
    """Solve an SMT2 file with a multi-strategy portfolio.

    Strategy order (budget split across strategies):
      1. SolverFor('QF_LIA') — DPLL(T) with LIA theory; empirically solves
         many bounded QF_NIA instances that the default nlsat-based strategy
         cannot handle within reasonable time.
      2. Default Solver() — falls back to Z3's auto-configured strategy.
    """
    wall_start = time.perf_counter()
    deadline = wall_start + timeout

    strategies = [
        ("QF_LIA", lambda: SolverFor('QF_LIA')),
        ("default", Solver),
    ]
    n = len(strategies)

    try:
        for idx, (tag, make_solver) in enumerate(strategies):
            remaining = deadline - time.perf_counter()
            if remaining < 1:
                break

            is_last = (idx == n - 1)
            budget_ms = remaining * 1000 if is_last else (remaining / (n - idx)) * 1000

            r = _try_solver(make_solver(), filepath, budget_ms)

            if r == sat:
                elapsed = time.perf_counter() - wall_start
                return {'status': 'SAT', 'objective': 'N/A', 'time': elapsed}
            if r == unsat:
                elapsed = time.perf_counter() - wall_start
                return {'status': 'UNSAT', 'objective': 'N/A', 'time': elapsed}

        elapsed = time.perf_counter() - wall_start
        if elapsed >= timeout * 0.95:
            return {'status': 'TIMEOUT', 'objective': 'N/A', 'time': elapsed}
        return {'status': 'UNKNOWN', 'objective': 'N/A', 'time': elapsed}

    except Exception as e:
        return {'status': 'ERROR', 'objective': 'N/A',
                'time': time.perf_counter() - wall_start,
                'error_msg': f'{type(e).__name__}: {e}'}


# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Z3 baseline solver")
    ap.add_argument("instance", help="Input file (QPLIB / SMT2 / CNF / JSON)")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--k", type=int, default=None,
                    help="Diverse-SAT: number of diverse models (k >= 2). "
                         "Only for .cnf inputs.")
    args = ap.parse_args()

    name = Path(args.instance).name

    if not HAS_Z3:
        print(f">>> Benchmark {name} Solver Z3 "
              f"Status ERROR OPT N/A TimeTotal 0.000")
        print("Z3 Python bindings not installed", file=sys.stderr)
        return

    start_wall = time.perf_counter()
    try:
        if args.instance.endswith('.smt2'):
            result = solve_smt2(args.instance, args.timeout)
        else:
            problem = load_instance(args.instance, k=args.k)
            result = solve_qplib(problem, args.timeout)
    except Exception as e:
        elapsed = time.perf_counter() - start_wall
        msg = f'{type(e).__name__}: {e}'
        # Z3 may raise "canceled" near timeout in nonlinear optimization loops.
        # Treat this as TIMEOUT instead of ERROR to avoid inflating error counts.
        if "canceled" in str(e).lower():
            result = {'status': 'TIMEOUT', 'objective': None, 'time': elapsed,
                      'error_msg': msg}
        else:
            result = {'status': 'ERROR', 'objective': None, 'time': elapsed,
                      'error_msg': msg}
        traceback.print_exc(file=sys.stderr)

    obj_str = str(result['objective']) if result.get('objective') is not None else 'N/A'
    status = result['status']

    print(f">>> Benchmark {name} Solver Z3 "
          f"Status {status} OPT {obj_str} TimeTotal {result.get('time', 0.0):.3f}")

    if result.get('error_msg'):
        print(f"Error: {result['error_msg']}", file=sys.stderr)


if __name__ == "__main__":
    main()
