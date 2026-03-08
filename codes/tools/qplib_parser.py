#!/usr/bin/env python3
import os
from typing import Dict, Any, List


def _strip_comment(line: str) -> str:
    return line.split('#')[0].strip()


def parse_qplib_file(filepath: str) -> Dict[str, Any]:

    with open(filepath, 'r') as f:
        raw_lines = [line.rstrip('\n') for line in f]

    pos = [0]

    def next_line() -> str:
        while pos[0] < len(raw_lines):
            line = raw_lines[pos[0]].strip()
            pos[0] += 1
            if line:
                return line
        return ''

    def next_val() -> str:
        return _strip_comment(next_line())

    def next_int() -> int:
        return int(float(next_val()))

    def next_float() -> float:
        return float(next_val())

    # ── 1. Header ──────────────────────────────────────────────
    problem_name = next_val()
    problem_type = next_val()
    obj_sense_raw = next_val()
    num_vars = next_int()

    obj_type = problem_type[0]   # Q / L / D / C / N
    var_type = problem_type[1]   # B / I / C / M / G
    con_type = problem_type[2]   # B / L / Q / N

    has_constraints = (con_type != 'B')
    has_obj_quad = (obj_type in ('Q', 'D', 'C'))
    has_con_quad = (con_type in ('Q', 'N'))
    is_binary = (var_type == 'B')

    num_constraints = next_int() if has_constraints else 0

    problem = {
        'name': problem_name,
        'type': problem_type,
        'source': os.path.abspath(filepath),
        'variables': {},
        'objective': {
            'sense': 'max' if obj_sense_raw == 'maximize' else 'min',
            'terms': []
        },
        'constraints': []
    }

    # ── 2. Objective quadratic terms ───────────────────────────
    if has_obj_quad:
        num_quad = next_int()
        for _ in range(num_quad):
            parts = next_line().split()
            i, j = int(parts[0]), int(parts[1])
            val = float(parts[2])
            if val == 0:
                continue
            if i == j:
                # Hessian convention: coeff of x_i^2 = Q[i][i] / 2
                problem['objective']['terms'].append({
                    'c': val / 2.0,
                    'vars': {f'x{i}': 2}
                })
            else:
                # Off-diagonal: coeff of x_i*x_j = Q[i][j]
                problem['objective']['terms'].append({
                    'c': val,
                    'vars': {f'x{i}': 1, f'x{j}': 1}
                })

    # ── 3. Objective linear terms (default + non-default) ─────
    default_linear = next_float()
    num_nd_linear = next_int()
    linear_overrides = {}
    for _ in range(num_nd_linear):
        parts = next_line().split()
        linear_overrides[int(parts[0])] = float(parts[1])

    for vi in range(1, num_vars + 1):
        c = linear_overrides.get(vi, default_linear)
        if c != 0:
            problem['objective']['terms'].append({
                'c': c,
                'vars': {f'x{vi}': 1}
            })

    # ── 4. Objective constant ─────────────────────────────────
    obj_constant = next_float()
    if obj_constant != 0:
        problem['objective']['terms'].append({
            'c': obj_constant,
            'vars': {}
        })

    # ── 5. Constraint quadratic terms (Hessians, if Q/N) ──────
    con_quad = {}
    if has_constraints and has_con_quad:
        num_cq = next_int()
        for _ in range(num_cq):
            parts = next_line().split()
            ci, i, j = int(parts[0]), int(parts[1]), int(parts[2])
            val = float(parts[3])
            if val == 0:
                continue
            con_quad.setdefault(ci, [])
            if i == j:
                con_quad[ci].append({'c': val / 2.0, 'vars': {f'x{i}': 2}})
            else:
                con_quad[ci].append({'c': val, 'vars': {f'x{i}': 1, f'x{j}': 1}})

    # ── 6. Constraint linear terms (Jacobian) ─────────────────
    con_linear = {}
    if has_constraints:
        num_jac = next_int()
        for _ in range(num_jac):
            parts = next_line().split()
            ci, vi = int(parts[0]), int(parts[1])
            val = float(parts[2])
            if val == 0:
                continue
            con_linear.setdefault(ci, [])
            con_linear[ci].append({'c': val, 'vars': {f'x{vi}': 1}})

    # ── 7. Bounds: infinity, LHS, RHS ─────────────────────────
    infinity_val = next_float()

    if has_constraints:
        default_lhs = next_float()
        num_nd_lhs = next_int()
        lhs_map = {}
        for _ in range(num_nd_lhs):
            parts = next_line().split()
            lhs_map[int(parts[0])] = float(parts[1])

        default_rhs = next_float()
        num_nd_rhs = next_int()
        rhs_map = {}
        for _ in range(num_nd_rhs):
            parts = next_line().split()
            rhs_map[int(parts[0])] = float(parts[1])

        # Build constraint list
        for ci in range(1, num_constraints + 1):
            terms = con_quad.get(ci, []) + con_linear.get(ci, [])
            lhs = lhs_map.get(ci, default_lhs)
            rhs = rhs_map.get(ci, default_rhs)

            inf_lhs = (lhs <= -1e300)
            inf_rhs = (rhs >= 1e300)

            if inf_lhs and inf_rhs:
                continue
            elif inf_lhs:
                problem['constraints'].append({
                    'terms': terms, 'rel': '<=', 'rhs': rhs
                })
            elif inf_rhs:
                problem['constraints'].append({
                    'terms': terms, 'rel': '>=', 'rhs': lhs
                })
            elif abs(lhs - rhs) < 1e-10:
                problem['constraints'].append({
                    'terms': terms, 'rel': '==', 'rhs': rhs
                })
            else:
                problem['constraints'].append({
                    'terms': list(terms), 'rel': '>=', 'rhs': lhs
                })
                import copy
                problem['constraints'].append({
                    'terms': copy.deepcopy(terms), 'rel': '<=', 'rhs': rhs
                })

    # ── 8. Variable bounds ────────────────────────────────────
    if is_binary:
        for vi in range(1, num_vars + 1):
            problem['variables'][f'x{vi}'] = {'lb': 0, 'ub': 1}
    else:
        default_lb = next_float()
        num_nd_lb = next_int()
        lb_map = {}
        for _ in range(num_nd_lb):
            parts = next_line().split()
            lb_map[int(parts[0])] = float(parts[1])

        default_ub = next_float()
        num_nd_ub = next_int()
        ub_map = {}
        for _ in range(num_nd_ub):
            parts = next_line().split()
            ub_map[int(parts[0])] = float(parts[1])

        for vi in range(1, num_vars + 1):
            lb = lb_map.get(vi, default_lb)
            ub = ub_map.get(vi, default_ub)
            if lb <= -1e300:
                lb = 0
            if ub >= 1e300:
                ub = 1000
            problem['variables'][f'x{vi}'] = {
                'lb': int(lb), 'ub': int(ub)
            }

    print(f"Parsed {problem_name} ({problem_type}): "
          f"{num_vars} vars, {num_constraints} constrs, "
          f"{len(problem['objective']['terms'])} obj terms")

    return problem
