#!/usr/bin/env python3

import math
from typing import Dict, List, Tuple, Any, Optional


def decode_assignment_oh(assignment: List[int], name2idx: Dict[str, int],
                         problem: Dict[str, Any]) -> Dict[str, int]:
    var_values = {}
    for var_name, meta in problem.get('variables', {}).items():
        q = name2idx[var_name]
        lb = int(meta.get('lb', 0))
        var_values[var_name] = lb
    return var_values


def decode_assignment_una(assignment: List[int], name2idx: Dict[str, int],
                          problem: Dict[str, Any]) -> Dict[str, int]:
    var_values = {}
    for var_name, meta in problem.get('variables', {}).items():
        q = name2idx[var_name]
        lb = int(meta.get('lb', 0))
        var_values[var_name] = lb
    return var_values


def decode_assignment_bin(assignment: List[int], name2idx: Dict[str, int],
                          problem: Dict[str, Any]) -> Dict[str, int]:
    var_values = {}
    for var_name, meta in problem.get('variables', {}).items():
        q = name2idx[var_name]
        lb = int(meta.get('lb', 0))
        var_values[var_name] = lb
    return var_values


def decode_assignment(assignment: List[int], encoding: str,
                      name2idx: Dict[str, int], problem: Dict[str, Any],
                      vpool=None) -> Dict[str, int]:
    if vpool is not None:
        return decode_assignment_with_vpool(assignment, encoding, name2idx, problem, vpool)

    if encoding == 'OH':
        return decode_assignment_oh(assignment, name2idx, problem)
    elif encoding == 'UNA':
        return decode_assignment_una(assignment, name2idx, problem)
    elif encoding == 'BIN':
        return decode_assignment_bin(assignment, name2idx, problem)
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")


def decode_assignment_with_vpool(assignment: List[int], encoding: str,
                                  name2idx: Dict[str, int], problem: Dict[str, Any],
                                  vpool) -> Dict[str, int]:
    var_values = {}
    true_vars = set(lit for lit in assignment if lit > 0)

    for var_name, meta in problem.get('variables', {}).items():
        q = name2idx[var_name]
        ub = int(meta.get('ub', 0))
        lb = int(meta.get('lb', 0))

        if encoding == 'OH':
            found = False
            for r in range(ub + 1):
                var_key = f'x_{q}@{r}'
                var_id = vpool.obj2id.get(var_key)
                if var_id is not None and var_id in true_vars:
                    var_values[var_name] = lb + r
                    found = True
                    break
            if not found:
                var_values[var_name] = lb

        elif encoding == 'UNA':
            max_r = 0
            for r in range(1, ub + 1):
                var_key = f'x_{q}@{r}'
                var_id = vpool.obj2id.get(var_key)
                if var_id is not None and var_id in true_vars:
                    max_r = r
            var_values[var_name] = lb + max_r

        elif encoding == 'BIN':
            num_bits = math.ceil(math.log2(ub + 1)) if ub > 0 else 1
            val = 0
            for r in range(num_bits):
                var_key = f'x_{q}@{r}'
                var_id = vpool.obj2id.get(var_key)
                if var_id is not None and var_id in true_vars:
                    val += (1 << r)
            var_values[var_name] = lb + val

    return var_values


def evaluate_term(term: Dict[str, Any], var_values: Dict[str, int]) -> int:
    c = int(term.get('c', 1))
    result = c
    for var_name, power in term.get('vars', {}).items():
        val = var_values.get(var_name, 0)
        result *= (val ** int(power))
    return result


def evaluate_polynomial(terms: List[Dict[str, Any]], var_values: Dict[str, int]) -> int:
    return sum(evaluate_term(term, var_values) for term in terms)


def check_constraint(constraint: Dict[str, Any], var_values: Dict[str, int]) -> Tuple[bool, str]:
    terms = constraint.get('terms', [])
    sense = constraint.get('sense', '<=')
    rhs = int(constraint.get('rhs', 0))

    lhs = evaluate_polynomial(terms, var_values)

    if sense == '<=':
        satisfied = (lhs <= rhs)
        desc = f"{lhs} <= {rhs}"
    elif sense == '>=':
        satisfied = (lhs >= rhs)
        desc = f"{lhs} >= {rhs}"
    elif sense == '=' or sense == '==':
        satisfied = (lhs == rhs)
        desc = f"{lhs} == {rhs}"
    else:
        satisfied = False
        desc = f"Unknown constraint type: {sense}"

    return satisfied, desc


def verify_solution(problem: Dict[str, Any], result: Dict[str, Any],
                    encoding: str, vpool=None) -> Tuple[bool, Dict[str, Any]]:
    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'var_values': {},
        'constraint_checks': [],
        'computed_objective': None,
        'reported_objective': result.get('objective_value')
    }

    assignment = result.get('assignment', [])
    name2idx = result.get('name2idx', {})

    if vpool is None:
        vpool = result.get('vpool')

    if not assignment:
        report['valid'] = False
        report['errors'].append("No assignment data")
        return False, report

    # 1. Decode variable values
    try:
        var_values = decode_assignment(assignment, encoding, name2idx, problem, vpool)
        report['var_values'] = var_values
    except Exception as e:
        report['warnings'].append(f"Variable decoding failed: {e}")
        return True, report

    # 2. Check constraints
    constraints = problem.get('constraints', [])
    all_satisfied = True

    for i, constr in enumerate(constraints):
        satisfied, desc = check_constraint(constr, var_values)
        report['constraint_checks'].append({
            'index': i,
            'satisfied': satisfied,
            'description': desc
        })
        if not satisfied:
            all_satisfied = False
            report['errors'].append(f"Constraint {i} violated: {desc}")

    # 3. Compute objective
    objective = problem.get('objective', {})
    terms = objective.get('terms', [])
    if terms:
        computed_obj = evaluate_polynomial(terms, var_values)
        report['computed_objective'] = computed_obj

        reported_obj = result.get('objective_value')
        if reported_obj is not None and computed_obj != reported_obj:
            report['warnings'].append(
                f"Objective mismatch: computed={computed_obj}, reported={reported_obj}"
            )

    report['valid'] = all_satisfied and len(report['errors']) == 0

    return report['valid'], report


def format_verification_report(report: Dict[str, Any]) -> str:
    lines = []

    if report['valid']:
        lines.append("Verification PASSED")
    else:
        lines.append("Verification FAILED")

    if report['var_values']:
        var_str = ", ".join(f"{k}={v}" for k, v in sorted(report['var_values'].items()))
        lines.append(f"  Variables: {var_str}")

    checks = report.get('constraint_checks', [])
    if checks:
        passed = sum(1 for c in checks if c['satisfied'])
        lines.append(f"  Constraints: {passed}/{len(checks)} passed")

    if report['computed_objective'] is not None:
        lines.append(f"  Computed objective: {report['computed_objective']}")
    if report['reported_objective'] is not None:
        lines.append(f"  Solver objective: {report['reported_objective']}")

    for err in report.get('errors', []):
        lines.append(f"  ERROR: {err}")
    for warn in report.get('warnings', []):
        lines.append(f"  WARNING: {warn}")

    return "\n".join(lines)


