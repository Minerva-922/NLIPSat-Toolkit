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
    import cplex
    from cplex.exceptions import CplexError
    HAS_CPLEX = True
except ImportError:
    HAS_CPLEX = False


def load_instance(filepath):
    if filepath.endswith('.qplib'):
        from tools.qplib_parser import parse_qplib_file
        return parse_qplib_file(filepath)
    elif filepath.endswith('.smt2'):
        from tools.smt2_parser import parse_smt2_file
        return parse_smt2_file(filepath)
    else:
        with open(filepath, 'r') as f:
            return json.load(f)


def check_max_degree(problem):
    max_deg = 0
    for term in problem.get('objective', {}).get('terms', []):
        deg = sum(int(v) for v in term.get('vars', {}).values())
        max_deg = max(max_deg, deg)
    for constr in problem.get('constraints', []):
        for term in constr.get('terms', []):
            deg = sum(int(v) for v in term.get('vars', {}).values())
            max_deg = max(max_deg, deg)
    return max_deg


def solve_with_cplex(instance_path, timeout=7200):
    if not HAS_CPLEX:
        return {'status': 'ERROR', 'objective': None, 'time': 0}

    start_time = time.perf_counter()
    try:
        problem = load_instance(instance_path)

        max_deg = check_max_degree(problem)
        if max_deg > 2:
            return {
                'status': 'UNSUPPORTED',
                'objective': None,
                'time': time.perf_counter() - start_time,
            }

        model = cplex.Cplex()
        model.set_log_stream(None)
        model.set_error_stream(None)
        model.set_warning_stream(None)
        model.set_results_stream(None)
        # Use wall-clock timing for fair timeout accounting.
        model.parameters.clocktype.set(1)
        model.parameters.timelimit.set(timeout)
        # Single thread for fair comparison with SAT/MaxSAT baselines.
        model.parameters.threads.set(1)
        # Tighten tolerances to avoid reporting near-integral/near-optimal
        # solutions as exact optima under loose defaults.
        model.parameters.mip.tolerances.integrality.set(0.0)
        model.parameters.mip.tolerances.mipgap.set(0.0)
        model.parameters.mip.tolerances.absmipgap.set(0.0)
        model.parameters.simplex.tolerances.optimality.set(1e-9)
        # Enable global optimality for non-convex QP/MIQP (fixes QIL instances)
        model.parameters.optimalitytarget.set(3)

        sense = problem.get('objective', {}).get('sense', 'max')
        if sense == 'max':
            model.objective.set_sense(model.objective.sense.maximize)
        else:
            model.objective.set_sense(model.objective.sense.minimize)

        var_names = []
        var_lbs = []
        var_ubs = []
        for var_name, info in problem.get('variables', {}).items():
            var_names.append(var_name)
            var_lbs.append(float(info.get('lb', 0)))
            var_ubs.append(float(info.get('ub', 100)))

        model.variables.add(
            names=var_names,
            lb=var_lbs,
            ub=var_ubs,
            types=['I'] * len(var_names),
        )

        linear_obj = {name: 0.0 for name in var_names}
        quad_coeffs = []
        obj_offset = 0.0

        for term in problem.get('objective', {}).get('terms', []):
            c = float(term.get('c', 1))
            tvars = term.get('vars', {})
            if not tvars:
                obj_offset += c
                continue
            var_list = list(tvars.keys())
            powers = [int(tvars[v]) for v in var_list]
            total_degree = sum(powers)

            if total_degree == 1:
                linear_obj[var_list[0]] += c
            elif total_degree == 2:
                if len(var_list) == 1 and powers[0] == 2:
                    # CPLEX uses 0.5 * x^T Q x convention:
                    # diagonal Q[i][i] = 2*c so that 0.5 * 2*c * x_i^2 = c * x_i^2
                    quad_coeffs.append((var_list[0], var_list[0], 2 * c))
                elif len(var_list) == 2:
                    # Off-diagonal: CPLEX does NOT auto-symmetrize Q.
                    # To get coefficient c for x_i*x_j, we need Q[i][j] + Q[j][i] = 2*c
                    # so that 0.5 * 2*c * x_i*x_j = c * x_i*x_j.
                    # Setting one entry to 2*c achieves this.
                    quad_coeffs.append((var_list[0], var_list[1], c))

        model.objective.set_linear(
            [(name, linear_obj[name]) for name in var_names]
        )
        if quad_coeffs:
            model.objective.set_quadratic_coefficients(quad_coeffs)
        if obj_offset != 0:
            model.objective.set_offset(obj_offset)

        for idx, constr in enumerate(problem.get('constraints', [])):
            lin_expr = []
            quad_expr = []
            for term in constr.get('terms', []):
                c = float(term.get('c', 1))
                tvars = term.get('vars', {})
                var_list = list(tvars.keys())
                powers = [int(tvars[v]) for v in var_list]
                total_degree = sum(powers)

                if total_degree == 1 and len(var_list) == 1:
                    lin_expr.append((var_list[0], c))
                elif total_degree == 2:
                    if len(var_list) == 1 and powers[0] == 2:
                        # x_i^2: CPLEX uses 0.5*x^T*Q*x, so Q[i][i] = 2*c
                        quad_expr.append((var_list[0], var_list[0], 2 * c))
                    elif len(var_list) == 2:
                        # x_i*x_j: need 2*c for the same 0.5*Q convention
                        quad_expr.append((var_list[0], var_list[1], c))

            if not lin_expr and not quad_expr:
                continue

            rel = constr.get('rel', constr.get('sense', '<='))
            rhs = float(constr.get('rhs', 0))
            sense_map = {'<=': 'L', '>=': 'G', '==': 'E', '=': 'E'}

            if quad_expr:
                # Quadratic constraint: use CPLEX quadratic_constraints.add
                lin_ind = [v for v, _ in lin_expr]
                lin_val = [c for _, c in lin_expr]
                quad_ind1 = [q[0] for q in quad_expr]
                quad_ind2 = [q[1] for q in quad_expr]
                quad_val = [q[2] for q in quad_expr]

                model.quadratic_constraints.add(
                    lin_expr=cplex.SparsePair(ind=lin_ind, val=lin_val) if lin_expr else cplex.SparsePair(ind=[], val=[]),
                    quad_expr=cplex.SparseTriple(ind1=quad_ind1, ind2=quad_ind2, val=quad_val),
                    sense=sense_map.get(rel, 'L'),
                    rhs=rhs,
                    name=f"c{idx}",
                )
            else:
                # Pure linear constraint
                model.linear_constraints.add(
                    lin_expr=[cplex.SparsePair(
                        ind=[v for v, _ in lin_expr],
                        val=[c for _, c in lin_expr],
                    )],
                    senses=[sense_map.get(rel, 'L')],
                    rhs=[rhs],
                    names=[f"c{idx}"],
                )

        model.solve()
        solve_time = time.perf_counter() - start_time

        status_str = model.solution.get_status_string()
        try:
            obj_value = model.solution.get_objective_value()
        except CplexError:
            obj_value = None

        cplex_status = "OPTIMAL" if "optimal" in status_str.lower() else status_str
        return {'status': cplex_status, 'objective': obj_value, 'time': solve_time}

    except Exception as e:
        return {
            'status': 'ERROR',
            'objective': None,
            'time': time.perf_counter() - start_time,
            'error_msg': str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="CPLEX baseline solver")
    parser.add_argument("instance", help="Input instance (JSON / QPLIB)")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    if not HAS_CPLEX:
        print(f">>> Benchmark {Path(args.instance).name} Solver CPLEX "
              f"Status ERROR OPT N/A TimeTotal 0.000")
        return

    result = solve_with_cplex(args.instance, args.timeout)
    opt_str = f"{result['objective']:.6f}" if result['objective'] is not None else "N/A"

    print(f">>> Benchmark {Path(args.instance).name} Solver CPLEX "
          f"Status {result['status']} OPT {opt_str} TimeTotal {result['time']:.3f}")
    if result.get('error_msg'):
        print(f"Error detail: {result['error_msg']}", file=sys.stderr)


if __name__ == "__main__":
    main()
