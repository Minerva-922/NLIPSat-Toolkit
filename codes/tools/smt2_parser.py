#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

try:
    import z3
except ImportError:
    raise ImportError("z3-solver required. Install: pip install z3-solver")


@dataclass
class ParseStats:
    total_assertions: int = 0
    bound_constraints: int = 0
    polynomial_constraints: int = 0
    equality_definitions: int = 0
    and_constraints: int = 0
    or_constraints: int = 0
    skipped_constraints: int = 0
    skipped_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    max_degree: int = 0

    def summary(self) -> str:
        lines = [
            f"Total assertions: {self.total_assertions}",
            f"Bound constraints: {self.bound_constraints}",
            f"Polynomial constraints: {self.polynomial_constraints}",
            f"Equality definitions (var=poly): {self.equality_definitions}",
            f"AND constraints (expanded): {self.and_constraints}",
            f"OR constraints: {self.or_constraints}",
            f"Skipped constraints: {self.skipped_constraints}",
            f"Max polynomial degree: {self.max_degree}",
        ]
        if self.skipped_reasons:
            lines.append("Skipped reasons:")
            for reason, count in sorted(self.skipped_reasons.items(), key=lambda x: -x[1]):
                lines.append(f"  - {reason}: {count}")
        return "\n".join(lines)


def parse_smt2_file(filepath: str, objective_mode: str = "sum",
                    verbose: bool = False, default_ub: int = 1000) -> Dict[str, Any]:
    path = Path(filepath)
    stats = ParseStats()

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # 1. Extract variable declarations via regex
    variables = {}
    int_var_names = []
    bool_var_names = []

    decl_pattern = r'\(declare-(?:fun|const)\s+(\|[^|]*\||[^\s)]+)(?:\s*\(\))?\s+(Int|Bool)\)'
    for match in re.finditer(decl_pattern, content):
        name, var_type = match.groups()
        clean_name = name.strip('|')
        if var_type == 'Int':
            int_var_names.append(clean_name)
            variables[clean_name] = {'lb': None, 'ub': None}
        elif var_type == 'Bool':
            bool_var_names.append(clean_name)
            variables[clean_name] = {'lb': 0, 'ub': 1}

    # 2. Use Z3 to extract bounds and constraints
    constraints = []
    try:
        assertions = z3.parse_smt2_file(filepath)
        stats.total_assertions = len(assertions)

        _extract_bounds_z3(assertions, variables, int_var_names, stats)
        all_constraints = _extract_all_constraints_z3(assertions, int_var_names, stats,
                                                       variables=variables)
        constraints.extend(all_constraints)

    except Exception as e:
        if verbose:
            print(f"Warning: Z3 parse failed: {e}")

    # 3. Set default bounds for variables without inferred bounds
    for name in int_var_names:
        if variables[name]['lb'] is None:
            variables[name]['lb'] = 0
        if variables[name]['ub'] is None:
            variables[name]['ub'] = default_ub

    # 4. Build objective function
    objective = _build_objective(int_var_names, objective_mode)

    # 5. Update max degree from objective terms
    for term in objective['terms']:
        degree = sum(term['vars'].values()) if term['vars'] else 0
        stats.max_degree = max(stats.max_degree, degree)

    # 6. Normalize variable names (SMT2 names may contain special chars)
    name_map = {}
    normalized_vars = {}
    for i, name in enumerate(sorted(variables.keys())):
        safe_name = f"x{i+1}"
        name_map[name] = safe_name
        normalized_vars[safe_name] = variables[name]

    normalized_constraints = []
    for c in constraints:
        new_terms = []
        for term in c['terms']:
            new_vars = {name_map.get(v, v): exp for v, exp in term['vars'].items()}
            new_terms.append({'c': term['c'], 'vars': new_vars})
        normalized_constraints.append({
            'terms': new_terms,
            'rel': c['rel'],
            'rhs': c['rhs']
        })

    normalized_obj_terms = []
    for term in objective['terms']:
        new_vars = {name_map.get(v, v): exp for v, exp in term['vars'].items()}
        normalized_obj_terms.append({'c': term['c'], 'vars': new_vars})

    problem = {
        'name': path.stem,
        'type': 'SMT2_NIA',
        'source': str(path),
        'variables': normalized_vars,
        'objective': {
            'sense': objective['sense'],
            'terms': normalized_obj_terms
        },
        'constraints': normalized_constraints,
        '_name_map': name_map,
        '_parse_stats': {
            'total_assertions': stats.total_assertions,
            'polynomial_constraints': stats.polynomial_constraints,
            'equality_definitions': stats.equality_definitions,
            'max_degree': stats.max_degree,
            'skipped': stats.skipped_constraints
        }
    }

    if verbose:
        print(f"\n=== Parse Stats for {path.name} ===")
        print(stats.summary())

    return problem


def _extract_bounds_z3(assertions, variables: Dict, int_var_names: List[str], stats: ParseStats):
    queue = list(assertions)

    while queue:
        expr = queue.pop(0)

        if z3.is_and(expr):
            queue.extend(expr.children())
            continue

        if not z3.is_app(expr) or len(expr.children()) != 2:
            continue

        kind = expr.decl().kind()
        lhs, rhs = expr.children()

        bound_extracted = False

        # Form: var rel const
        if _is_int_var(lhs) and z3.is_int_value(rhs):
            var_name = _get_var_name(lhs)
            if var_name in variables:
                _update_bound(variables, var_name, kind, rhs.as_long(), is_rhs_const=True)
                bound_extracted = True

        # Form: const rel var
        elif z3.is_int_value(lhs) and _is_int_var(rhs):
            var_name = _get_var_name(rhs)
            if var_name in variables:
                _update_bound(variables, var_name, kind, lhs.as_long(), is_rhs_const=False)
                bound_extracted = True

        if bound_extracted:
            stats.bound_constraints += 1


def _get_var_name(expr) -> str:
    name = expr.decl().name()
    return name.strip('|')


def _is_int_var(expr) -> bool:
    return (z3.is_const(expr) and
            expr.decl().kind() == z3.Z3_OP_UNINTERPRETED and
            expr.sort().kind() == z3.Z3_INT_SORT)


def _update_bound(variables: Dict, var_name: str, kind: int, val: int, is_rhs_const: bool):
    lb = variables[var_name]['lb']
    ub = variables[var_name]['ub']

    if kind == z3.Z3_OP_EQ:
        lb = max(lb, val) if lb is not None else val
        ub = min(ub, val) if ub is not None else val
    elif kind == z3.Z3_OP_LE:
        if is_rhs_const:
            ub = min(ub, val) if ub is not None else val
        else:
            lb = max(lb, val) if lb is not None else val
    elif kind == z3.Z3_OP_GE:
        if is_rhs_const:
            lb = max(lb, val) if lb is not None else val
        else:
            ub = min(ub, val) if ub is not None else val
    elif kind == z3.Z3_OP_LT:
        if is_rhs_const:
            ub = min(ub, val - 1) if ub is not None else val - 1
        else:
            lb = max(lb, val + 1) if lb is not None else val + 1
    elif kind == z3.Z3_OP_GT:
        if is_rhs_const:
            lb = max(lb, val + 1) if lb is not None else val + 1
        else:
            ub = min(ub, val - 1) if ub is not None else val - 1

    variables[var_name]['lb'] = lb
    variables[var_name]['ub'] = ub


def _extract_all_constraints_z3(assertions, int_var_names: List[str], stats: ParseStats,
                                 variables: Dict = None) -> List[Dict]:
    constraints = []
    queue = list(assertions)

    while queue:
        expr = queue.pop(0)

        if z3.is_and(expr):
            queue.extend(expr.children())
            stats.and_constraints += 1
            continue

        if z3.is_or(expr):
            or_result = _handle_or_constraint(expr, int_var_names, stats, variables)
            if or_result is not None:
                constraints.extend(or_result)
                stats.or_constraints += 1
                continue
            stats.or_constraints += 1
            stats.skipped_reasons["OR constraint (complex)"] += 1
            stats.skipped_constraints += 1
            continue

        if z3.is_app(expr) and expr.decl().kind() == z3.Z3_OP_NOT:
            not_result = _handle_not_constraint(expr, int_var_names, stats, variables)
            if not_result is not None:
                constraints.extend(not_result)
                continue

        parsed = _parse_polynomial_constraint_general(expr, int_var_names, stats)
        if parsed:
            constraints.append(parsed)
            for term in parsed['terms']:
                degree = sum(term['vars'].values()) if term['vars'] else 0
                stats.max_degree = max(stats.max_degree, degree)

    return constraints


_not_and_counter = [0]


def _handle_not_constraint(expr, int_var_names: List[str], stats: ParseStats,
                           variables: Dict = None) -> Optional[List[Dict]]:
    """Handle NOT(AND(C_1, ..., C_n)) = OR(neg C_1, ..., neg C_n).

    For equalities C_j: f_j = 0 with companion f_j >= 0, the negation
    neg C_j simplifies to f_j >= 1.  We introduce auxiliary binary variables
    delta_j and encode:
      f_j >= delta_j  (i.e.  f_j - delta_j >= 0)   for each j
      delta_1 + ... + delta_n >= 1                   (at least one strict)
    """
    child = expr.children()[0]
    if not z3.is_and(child):
        return None

    sub_exprs = child.children()
    parsed_subs = []
    for se in sub_exprs:
        p = _parse_polynomial_constraint_general(se, int_var_names, stats)
        if p is None:
            return None
        parsed_subs.append(p)

    if not parsed_subs:
        return None

    batch_id = _not_and_counter[0]
    _not_and_counter[0] += 1

    result_constraints = []
    delta_names = []

    for j, sub in enumerate(parsed_subs):
        delta_name = f"__delta_{batch_id}_{j}"
        delta_names.append(delta_name)
        if variables is not None:
            variables[delta_name] = {'lb': 0, 'ub': 1}

        if sub['rel'] in ('==', '='):
            new_terms = [t.copy() for t in sub['terms']]
            new_terms.append({'c': -1.0, 'vars': {delta_name: 1}})
            result_constraints.append({
                'terms': new_terms,
                'rel': '>=',
                'rhs': sub['rhs']
            })
        elif sub['rel'] == '>=':
            neg_terms = [{'c': -t['c'], 'vars': dict(t['vars'])} for t in sub['terms']]
            neg_terms.append({'c': -1.0, 'vars': {delta_name: 1}})
            result_constraints.append({
                'terms': neg_terms,
                'rel': '>=',
                'rhs': -sub['rhs']
            })
        elif sub['rel'] == '<=':
            new_terms = [t.copy() for t in sub['terms']]
            new_terms.append({'c': -1.0, 'vars': {delta_name: 1}})
            result_constraints.append({
                'terms': new_terms,
                'rel': '>=',
                'rhs': sub['rhs']
            })
        else:
            return None

    alo_terms = [{'c': 1.0, 'vars': {dn: 1}} for dn in delta_names]
    result_constraints.append({
        'terms': alo_terms,
        'rel': '>=',
        'rhs': 1
    })

    stats.polynomial_constraints += len(result_constraints)
    return result_constraints


def _handle_or_constraint(expr, int_var_names: List[str], stats: ParseStats,
                          variables: Dict = None) -> Optional[List[Dict]]:
    children = expr.children()

    flat_subs = []
    for ch in children:
        if z3.is_not(ch):
            inner = ch.children()[0]
            p = _parse_polynomial_constraint_general(inner, int_var_names, stats)
            if p is None:
                return None
            if p['rel'] in ('==', '='):
                flat_subs.append(('neq', p))
            elif p['rel'] == '>=':
                flat_subs.append(('lt', p))
            elif p['rel'] == '<=':
                flat_subs.append(('gt', p))
            else:
                return None
        else:
            p = _parse_polynomial_constraint_general(ch, int_var_names, stats)
            if p is None:
                return None
            flat_subs.append(('pos', p))

    if not flat_subs:
        return None

    batch_id = _not_and_counter[0]
    _not_and_counter[0] += 1

    result_constraints = []
    sel_names = []

    for j, (kind, sub) in enumerate(flat_subs):
        sel_name = f"__sel_{batch_id}_{j}"
        sel_names.append(sel_name)
        if variables is not None:
            variables[sel_name] = {'lb': 0, 'ub': 1}

        if kind == 'pos':
            if sub['rel'] in ('>=', '>'):
                new_terms = [t.copy() for t in sub['terms']]
                new_terms.append({'c': -1.0, 'vars': {sel_name: 1}})
                result_constraints.append({
                    'terms': new_terms,
                    'rel': '>=',
                    'rhs': sub['rhs'] - 1
                })
            elif sub['rel'] in ('<=', '<'):
                neg_terms = [{'c': -t['c'], 'vars': dict(t['vars'])} for t in sub['terms']]
                neg_terms.append({'c': -1.0, 'vars': {sel_name: 1}})
                result_constraints.append({
                    'terms': neg_terms,
                    'rel': '>=',
                    'rhs': -sub['rhs'] - 1
                })
            else:
                return None
        elif kind == 'neq':
            new_terms = [t.copy() for t in sub['terms']]
            new_terms.append({'c': -1.0, 'vars': {sel_name: 1}})
            result_constraints.append({
                'terms': new_terms,
                'rel': '>=',
                'rhs': sub['rhs']
            })
        elif kind == 'lt':
            neg_terms = [{'c': -t['c'], 'vars': dict(t['vars'])} for t in sub['terms']]
            neg_terms.append({'c': -1.0, 'vars': {sel_name: 1}})
            result_constraints.append({
                'terms': neg_terms,
                'rel': '>=',
                'rhs': -sub['rhs']
            })
        elif kind == 'gt':
            new_terms = [t.copy() for t in sub['terms']]
            new_terms.append({'c': -1.0, 'vars': {sel_name: 1}})
            result_constraints.append({
                'terms': new_terms,
                'rel': '>=',
                'rhs': sub['rhs']
            })
        else:
            return None

    alo_terms = [{'c': 1.0, 'vars': {sn: 1}} for sn in sel_names]
    result_constraints.append({
        'terms': alo_terms,
        'rel': '>=',
        'rhs': 1
    })

    stats.polynomial_constraints += len(result_constraints)
    return result_constraints


def _parse_polynomial_constraint_general(expr, int_var_names: List[str], stats: ParseStats) -> Optional[Dict]:
    if not z3.is_app(expr):
        return None

    kind = expr.decl().kind()

    if kind == z3.Z3_OP_NOT:
        stats.skipped_reasons["NOT constraint (atomic)"] += 1
        stats.skipped_constraints += 1
        return None

    if kind == z3.Z3_OP_IMPLIES:
        stats.skipped_reasons["IMPLIES constraint"] += 1
        stats.skipped_constraints += 1
        return None

    if kind not in [z3.Z3_OP_LE, z3.Z3_OP_GE, z3.Z3_OP_LT, z3.Z3_OP_GT, z3.Z3_OP_EQ]:
        stats.skipped_reasons[f"Unknown op kind {kind}"] += 1
        stats.skipped_constraints += 1
        return None

    if len(expr.children()) != 2:
        stats.skipped_reasons["Non-binary constraint"] += 1
        stats.skipped_constraints += 1
        return None

    lhs, rhs = expr.children()

    lhs_terms = _parse_polynomial_expr(lhs, int_var_names)
    rhs_terms = _parse_polynomial_expr(rhs, int_var_names)

    if lhs_terms is None:
        stats.skipped_reasons["LHS parse failed"] += 1
        stats.skipped_constraints += 1
        return None

    if rhs_terms is None:
        stats.skipped_reasons["RHS parse failed"] += 1
        stats.skipped_constraints += 1
        return None

    # Standard form: lhs - rhs rel 0 (move all terms to left, constants to right)
    combined_terms = []
    rhs_constant = 0

    for term in lhs_terms:
        if not term['vars']:
            rhs_constant -= term['c']
        else:
            combined_terms.append(term.copy())

    for term in rhs_terms:
        if not term['vars']:
            rhs_constant += term['c']
        else:
            found = False
            for ct in combined_terms:
                if ct['vars'] == term['vars']:
                    ct['c'] -= term['c']
                    found = True
                    break
            if not found:
                combined_terms.append({'c': -term['c'], 'vars': term['vars'].copy()})

    combined_terms = [t for t in combined_terms if abs(t['c']) > 1e-10]

    rel_map = {
        z3.Z3_OP_LE: '<=',
        z3.Z3_OP_GE: '>=',
        z3.Z3_OP_LT: '<',
        z3.Z3_OP_GT: '>',
        z3.Z3_OP_EQ: '=='
    }
    rel = rel_map.get(kind, '<=')

    if kind == z3.Z3_OP_EQ and _is_int_var(lhs):
        stats.equality_definitions += 1

    stats.polynomial_constraints += 1

    return {
        'terms': combined_terms,
        'rel': rel,
        'rhs': rhs_constant
    }


def _parse_polynomial_expr(expr, int_var_names: List[str]) -> Optional[List[Dict]]:
    # Integer constant
    if z3.is_int_value(expr):
        val = expr.as_long()
        return [{'c': float(val), 'vars': {}}]

    # Integer variable
    if _is_int_var(expr):
        name = _get_var_name(expr)
        return [{'c': 1.0, 'vars': {name: 1}}]

    if not z3.is_app(expr):
        return None

    kind = expr.decl().kind()
    children = expr.children()

    # Addition
    if kind == z3.Z3_OP_ADD:
        result = []
        for child in children:
            child_terms = _parse_polynomial_expr(child, int_var_names)
            if child_terms is None:
                return None
            result.extend(child_terms)
        return _merge_like_terms(result)

    # Subtraction
    if kind == z3.Z3_OP_SUB:
        if len(children) == 1:
            terms = _parse_polynomial_expr(children[0], int_var_names)
            if terms is None:
                return None
            for t in terms:
                t['c'] = -t['c']
            return terms
        elif len(children) == 2:
            left_terms = _parse_polynomial_expr(children[0], int_var_names)
            right_terms = _parse_polynomial_expr(children[1], int_var_names)
            if left_terms is None or right_terms is None:
                return None
            for t in right_terms:
                t['c'] = -t['c']
            return _merge_like_terms(left_terms + right_terms)
        return None

    # Unary negation
    if kind == z3.Z3_OP_UMINUS:
        if len(children) != 1:
            return None
        terms = _parse_polynomial_expr(children[0], int_var_names)
        if terms is None:
            return None
        for t in terms:
            t['c'] = -t['c']
        return terms

    # Multiplication
    if kind == z3.Z3_OP_MUL:
        return _parse_multiplication(children, int_var_names)

    # ITE (if-then-else) - not supported
    if kind == z3.Z3_OP_ITE:
        return None

    return None


def _parse_multiplication(children, int_var_names: List[str]) -> Optional[List[Dict]]:
    coeff = 1.0
    combined_vars = {}
    need_distribution = False

    for child in children:
        if z3.is_int_value(child):
            coeff *= child.as_long()
        elif _is_int_var(child):
            name = _get_var_name(child)
            combined_vars[name] = combined_vars.get(name, 0) + 1
        else:
            need_distribution = True
            break

    if not need_distribution:
        return [{'c': coeff, 'vars': combined_vars}]

    # Distributive expansion: (a+b)*(c+d) = a*c + a*d + b*c + b*d
    result_terms = [{'c': 1.0, 'vars': {}}]

    for child in children:
        child_terms = _parse_polynomial_expr(child, int_var_names)
        if child_terms is None:
            return None

        new_result = []
        for rt in result_terms:
            for ct in child_terms:
                new_term = {
                    'c': rt['c'] * ct['c'],
                    'vars': rt['vars'].copy()
                }
                for v, e in ct['vars'].items():
                    new_term['vars'][v] = new_term['vars'].get(v, 0) + e
                new_result.append(new_term)
        result_terms = new_result

    return _merge_like_terms(result_terms)


def _merge_like_terms(terms: List[Dict]) -> List[Dict]:
    merged = {}

    for term in terms:
        key = tuple(sorted(term['vars'].items()))
        if key in merged:
            merged[key]['c'] += term['c']
        else:
            merged[key] = {'c': term['c'], 'vars': dict(term['vars'])}

    result = [t for t in merged.values() if abs(t['c']) > 1e-10]
    return result if result else [{'c': 0.0, 'vars': {}}]


def _build_objective(int_var_names: List[str], mode: str) -> Dict:
    if mode == "zero":
        return {'sense': 'min', 'terms': []}

    if mode == "first" and int_var_names:
        return {
            'sense': 'max',
            'terms': [{'c': 1.0, 'vars': {int_var_names[0]: 1}}]
        }

    # Default: maximize sum of all variables
    terms = [{'c': 1.0, 'vars': {name: 1}} for name in int_var_names]
    return {'sense': 'max', 'terms': terms}


