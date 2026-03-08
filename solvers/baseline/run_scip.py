#!/usr/bin/env python3

import argparse
import json
import sys
import os
import time
from pathlib import Path

CODES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "codes")
sys.path.insert(0, CODES_DIR)

try:
    from pyscipopt import Model
    HAS_SCIP = True
except ImportError:
    HAS_SCIP = False


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
        with open(filepath, 'r') as f:
            return json.load(f)


def _build_expr(terms, scip_vars):
    """Build a PySCIPOpt expression from parsed polynomial terms."""
    expr = 0
    for term in terms:
        c = float(term.get('c', 1))
        tvars = term.get('vars', {})
        if not tvars:
            expr += c
            continue
        monomial = c
        for vname, power in tvars.items():
            if vname in scip_vars:
                for _ in range(int(power)):
                    monomial *= scip_vars[vname]
        expr += monomial
    return expr


_LIMIT_STATUSES = frozenset({
    'bestsollimit', 'sollimit', 'timelimit', 'nodelimit',
    'totalnodelimit', 'stallnodelimit', 'memlimit', 'gaplimit',
    'restartlimit', 'primallimit', 'duallimit', 'userinterrupt',
})


def solve_with_scip(instance_path, timeout=3600, k=None):
    if not HAS_SCIP:
        return {'status': 'ERROR', 'objective': None, 'time': 0}

    start_time = time.perf_counter()
    try:
        if instance_path.endswith('.smt2'):
            return {'status': 'UNSUPPORTED', 'objective': None,
                    'time': time.perf_counter() - start_time}

        problem = load_instance(instance_path, k=k)

        model = Model()
        model.hideOutput()
        model.setRealParam('limits/time', timeout)
        model.setIntParam('parallel/maxnthreads', 1)
        model.setRealParam('numerics/feastol', 1e-8)
        model.setRealParam('limits/gap', 0.0)
        model.setRealParam('limits/absgap', 0.0)

        sense = problem.get('objective', {}).get('sense', 'min')

        scip_vars = {}
        for var_name, info in problem.get('variables', {}).items():
            lb = float(info.get('lb', 0))
            ub = float(info.get('ub', 100))
            scip_vars[var_name] = model.addVar(
                name=var_name, vtype='I', lb=lb, ub=ub)

        obj_expr = _build_expr(problem.get('objective', {}).get('terms', []), scip_vars)
        obj_is_linear = all(
            sum(int(v) for v in t.get('vars', {}).values()) <= 1
            for t in problem.get('objective', {}).get('terms', [])
        )
        if obj_is_linear:
            model.setObjective(obj_expr,
                               sense='maximize' if sense == 'max' else 'minimize')
        else:
            # Epigraph reformulation per official PySCIPOpt docs/FAQ:
            # SCIP only supports linear objectives, so introduce aux variable.
            # See: pyscipopt.recipes.nonlinear.set_nonlinear_objective
            obj_aux = model.addVar(name='_obj_aux', vtype='C',
                                   lb=-float('inf'), ub=float('inf'))
            if sense == 'min':
                model.addCons(obj_aux >= obj_expr, name='_obj_link')
                model.setObjective(obj_aux, sense='minimize')
            else:
                model.addCons(obj_aux <= obj_expr, name='_obj_link')
                model.setObjective(obj_aux, sense='maximize')

        for idx, constr in enumerate(problem.get('constraints', [])):
            lhs_expr = _build_expr(constr.get('terms', []), scip_vars)
            rel = constr.get('rel', constr.get('sense', '<='))
            rhs = float(constr.get('rhs', 0))

            if rel == '<=':
                model.addCons(lhs_expr <= rhs, name=f"c{idx}")
            elif rel == '>=':
                model.addCons(lhs_expr >= rhs, name=f"c{idx}")
            elif rel in ('==', '='):
                model.addCons(lhs_expr == rhs, name=f"c{idx}")

        model.optimize()
        solve_time = time.perf_counter() - start_time

        scip_status = model.getStatus()

        if scip_status == 'optimal':
            return {'status': 'OPTIMAL', 'objective': model.getObjVal(),
                    'time': solve_time}
        elif scip_status in _LIMIT_STATUSES:
            if model.getNSols() > 0:
                return {'status': 'FEASIBLE', 'objective': model.getObjVal(),
                        'time': solve_time}
            return {'status': 'TIMEOUT', 'objective': None, 'time': solve_time}
        elif scip_status == 'infeasible':
            return {'status': 'UNSAT', 'objective': None, 'time': solve_time}
        elif scip_status in ('unbounded', 'inforunbd'):
            return {'status': 'UNBOUNDED', 'objective': None, 'time': solve_time}
        else:
            return {'status': 'UNKNOWN', 'objective': None, 'time': solve_time}

    except Exception as e:
        return {
            'status': 'ERROR',
            'objective': None,
            'time': time.perf_counter() - start_time,
            'error_msg': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="SCIP baseline solver")
    parser.add_argument("instance", help="Input instance (QPLIB / CNF / JSON / SMT2)")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--k", type=int, default=None,
                        help="Diverse-SAT: number of diverse models (k >= 2). "
                             "Only for .cnf inputs.")
    args = parser.parse_args()

    name = Path(args.instance).name

    if not HAS_SCIP:
        print(f">>> Benchmark {name} Solver SCIP "
              f"Status ERROR OPT N/A TimeTotal 0.000")
        print("PySCIPOpt not installed", file=sys.stderr)
        return

    result = solve_with_scip(args.instance, args.timeout, k=args.k)
    opt_str = f"{result['objective']}" if result.get('objective') is not None else "N/A"
    if isinstance(result.get('objective'), float):
        v = result['objective']
        opt_str = str(int(v)) if v == int(v) else f"{v:.6f}"

    print(f">>> Benchmark {name} Solver SCIP "
          f"Status {result['status']} OPT {opt_str} TimeTotal {result['time']:.3f}")
    if result.get('error_msg'):
        print(f"Error detail: {result['error_msg']}", file=sys.stderr)


if __name__ == "__main__":
    main()
