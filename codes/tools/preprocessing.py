#!/usr/bin/env python3
from typing import Dict, Any, List, Tuple, Optional
from fractions import Fraction
from math import gcd
from functools import reduce
import copy


def _safe_fraction(val: Any, denom_cap: int) -> Fraction:
    if isinstance(val, Fraction):
        return Fraction(val.numerator, val.denominator if val.denominator != 0 else 1)
    try:
        return Fraction(val).limit_denominator(denom_cap)
    except Exception:
        try:
            return Fraction(int(val), 1)
        except Exception:
            return Fraction(0, 1)


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)


def _lcm_many(nums: List[int]) -> int:
    if not nums:
        return 1
    return reduce(_lcm, nums, 1)


def _gcd_many(nums: List[int]) -> int:
    nums = [abs(n) for n in nums if n != 0]
    if not nums:
        return 1
    return reduce(gcd, nums)


def _binom(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    num = 1
    den = 1
    for i in range(1, k + 1):
        num *= (n - (k - i))
        den *= i
    return num // den


def _expand_term_by_shifts(term: Dict[str, Any], shift_map: Dict[str, int]) -> Tuple[List[Tuple[Fraction, Dict[str, int]]], Fraction]:
    c = _safe_fraction(term.get('c', 1), 10**6)
    var_pows: Dict[str, int] = {vn: int(ep) for vn, ep in term.get('vars', {}).items()}

    # Incremental expansion
    partial: List[Tuple[Fraction, Dict[str, int]]] = [(c, {})]
    for v, k in var_pows.items():
        d = int(shift_map.get(v, 0))
        new_partial: List[Tuple[Fraction, Dict[str, int]]] = []
        for coeff, vp in partial:
            # (Y + d)^k = sum_{j=0}^{k} C(k,j) * Y^j * d^(k-j)
            for j in range(0, k + 1):
                coef2 = coeff * Fraction(_binom(k, j), 1) * (Fraction(d, 1) ** (k - j))
                vp2 = dict(vp)
                if j > 0:
                    vp2[v] = vp2.get(v, 0) + j
                new_partial.append((coef2, vp2))
        partial = new_partial

    # Merge like terms
    acc: Dict[Tuple[Tuple[str, int], ...], Fraction] = {}
    const_sum = Fraction(0, 1)
    for coeff, vp in partial:
        if not vp:
            const_sum += coeff
        else:
            key = tuple(sorted((vn, exp) for vn, exp in vp.items() if exp != 0))
            if key:
                acc[key] = acc.get(key, Fraction(0, 1)) + coeff
            else:
                const_sum += coeff

    items: List[Tuple[Fraction, Dict[str, int]]] = []
    for key, coef in acc.items():
        if coef == 0:
            continue
        vp = {vn: exp for vn, exp in key}
        items.append((coef, vp))
    return items, const_sum


def _canonicalize_polynomial(terms: List[Dict[str, Any]], shift_map: Dict[str, int]) -> Tuple[List[Dict[str, Any]], Fraction]:
    items_total: Dict[Tuple[Tuple[str, int], ...], Fraction] = {}
    const_total = Fraction(0, 1)
    for t in terms or []:
        items, const_sum = _expand_term_by_shifts(t, shift_map)
        const_total += const_sum
        for coef, vp in items:
            key = tuple(sorted(vp.items()))
            items_total[key] = items_total.get(key, Fraction(0, 1)) + coef
    new_terms: List[Dict[str, Any]] = []
    for key, coef in items_total.items():
        if coef == 0:
            continue
        new_terms.append({'c': coef, 'vars': dict(key)})
    return new_terms, const_total


def _reduce_binary_powers(terms: List[Dict[str, Any]], variables: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:

    acc: Dict[Tuple[Tuple[str, int], ...], Fraction] = {}
    for t in terms or []:
        coeff = _safe_fraction(t.get('c', 0), 10**6)
        if coeff == 0:
            continue

        raw_vars = t.get('vars', {}) or {}
        reduced_vars: Dict[str, int] = {}
        for vn, power in raw_vars.items():
            p = int(power)
            if p <= 0:
                continue
            info = variables.get(vn, {}) or {}
            lb = int(info.get('lb', 0))
            ub = int(info.get('ub', 0))
            # x in {0,1} => x^k == x for all k>=1
            reduced_vars[vn] = 1 if (lb == 0 and ub == 1) else p

        key = tuple(sorted((vn, exp) for vn, exp in reduced_vars.items() if exp != 0))
        acc[key] = acc.get(key, Fraction(0, 1)) + coeff

    merged: List[Dict[str, Any]] = []
    for key, coef in acc.items():
        if coef == 0:
            continue
        merged.append({'c': coef, 'vars': dict(key)})
    return merged


def _integerize_polynomial(terms: List[Dict[str, Any]], rhs: Optional[Fraction], denom_cap: int, scale_limit: int) -> Tuple[List[Dict[str, Any]], Optional[int], int]:
    denoms: List[int] = []
    coeff_fracs: List[Fraction] = []
    for t in terms:
        fc = _safe_fraction(t.get('c', 0), denom_cap)
        coeff_fracs.append(fc)
        denoms.append(abs(fc.denominator))
    rhs_frac = None
    if rhs is not None:
        rhs_frac = _safe_fraction(rhs, denom_cap)
        denoms.append(abs(rhs_frac.denominator))

    lcm_den = _lcm_many([d for d in denoms if d != 0]) or 1

    use_rounding = False
    if lcm_den > scale_limit:
        abs_vals = [abs(float(fc)) for fc in coeff_fracs if float(fc) != 0]
        if rhs_frac is not None and float(rhs_frac) != 0:
            abs_vals.append(abs(float(rhs_frac)))
        if abs_vals:
            min_abs = min(abs_vals)
            needed = int(1.0 / min_abs) + 1 if min_abs > 0 else scale_limit
            lcm_den = min(needed, scale_limit)
        else:
            lcm_den = scale_limit
        use_rounding = True

    int_coeffs: List[int] = []
    for fc in coeff_fracs:
        val = fc * lcm_den
        if use_rounding:
            int_coeffs.append(round(float(val)))
        else:
            int_coeffs.append(int(val.numerator // (val.denominator if val.denominator != 0 else 1)))
    int_rhs = None
    if rhs_frac is not None:
        val = rhs_frac * lcm_den
        if use_rounding:
            int_rhs = round(float(val))
        else:
            int_rhs = int(val.numerator // (val.denominator if val.denominator != 0 else 1))

    # Overall GCD reduction (including rhs)
    gcd_all = _gcd_many(int_coeffs + ([int_rhs] if int_rhs is not None else []))
    gcd_all = gcd_all if gcd_all != 0 else 1
    int_coeffs = [c // gcd_all for c in int_coeffs]
    if int_rhs is not None:
        int_rhs //= gcd_all

    new_terms: List[Dict[str, Any]] = []
    for (orig_t, ic) in zip(terms, int_coeffs):
        if ic != 0:
            new_terms.append({'c': ic, 'vars': dict(orig_t.get('vars', {}))})
    return new_terms, int_rhs, (lcm_den, gcd_all)


def _probe_binary_variables(
    terms: List[Dict[str, Any]],
    variables: Dict[str, Dict[str, Any]],
    constraints: List[Dict[str, Any]]
) -> Dict[str, int]:

    if not terms:
        return {}

    linear_coeff: Dict[str, float] = {}
    cross_coeffs: Dict[str, List[float]] = {}

    for t in terms:
        c_val = t.get('c', 0)
        c_val = float(c_val) if not isinstance(c_val, (int, float)) else float(c_val)
        vars_dict = t.get('vars', {}) or {}
        var_names = [vn for vn, p in vars_dict.items() if int(p) > 0]

        if len(var_names) == 1 and int(vars_dict[var_names[0]]) == 1:
            linear_coeff[var_names[0]] = linear_coeff.get(var_names[0], 0.0) + c_val
        elif len(var_names) == 2:
            for vn in var_names:
                cross_coeffs.setdefault(vn, []).append(c_val)

    constrained_vars: set = set()
    for cons in constraints or []:
        for t in cons.get('terms', []) or []:
            for vn in (t.get('vars', {}) or {}):
                constrained_vars.add(vn)

    fixed: Dict[str, int] = {}
    for vn, info_v in variables.items():
        lb = int(info_v.get('lb', 0))
        ub = int(info_v.get('ub', 0))
        if lb != 0 or ub != 1:
            continue
        if vn in constrained_vars:
            continue

        lin = linear_coeff.get(vn, 0.0)
        crosses = cross_coeffs.get(vn, [])
        neg_cross_sum = sum(c for c in crosses if c < 0)
        pos_cross_sum = sum(c for c in crosses if c > 0)

        if lin + neg_cross_sum > 0:
            fixed[vn] = 1
        elif lin + pos_cross_sum < 0:
            fixed[vn] = 0

    return fixed


def _substitute_fixed_variables(
    terms: List[Dict[str, Any]],
    fixed: Dict[str, int]
) -> Tuple[List[Dict[str, Any]], float]:

    if not fixed:
        return terms, 0.0

    acc: Dict[Tuple, float] = {}
    added_constant = 0.0

    for t in terms or []:
        c_val = t.get('c', 0)
        c_val = float(c_val) if not isinstance(c_val, (int, float)) else float(c_val)
        vars_dict = dict(t.get('vars', {}) or {})

        skip = False
        for vn in list(vars_dict.keys()):
            if vn in fixed:
                val = fixed[vn]
                if val == 0:
                    skip = True
                    break
                else:
                    del vars_dict[vn]

        if skip:
            continue

        if not vars_dict:
            added_constant += c_val
        else:
            key = tuple(sorted(vars_dict.items()))
            acc[key] = acc.get(key, 0.0) + c_val

    new_terms = []
    for key, coef in acc.items():
        if coef != 0:
            new_terms.append({'c': int(coef) if coef == int(coef) else coef,
                              'vars': dict(key)})
    return new_terms, added_constant


def _bound_tightening(
    constraints: List[Dict[str, Any]],
    variables: Dict[str, Dict[str, Any]],
    max_rounds: int = 10
) -> int:
    """Constraint propagation to tighten variable domains.
    For each linear constraint sum(a_i * x_i) rel rhs,
    derive tighter bounds on each variable."""
    if not constraints or not variables:
        return 0

    total_tightened = 0
    for _ in range(max_rounds):
        changed = False
        for cons in constraints:
            terms = cons.get('terms', []) or []
            rel = cons.get('sense', cons.get('rel', '<='))
            rhs_val = cons.get('rhs', 0)
            if rhs_val is None:
                continue
            try:
                rhs = float(rhs_val)
            except (TypeError, ValueError):
                continue

            linear_terms = []
            for t in terms:
                vs = t.get('vars', {}) or {}
                if sum(int(p) for p in vs.values()) != 1:
                    continue
                vn = list(vs.keys())[0]
                c_val = t.get('c', 0)
                try:
                    c_val = float(c_val) if not isinstance(c_val, (int, float)) else float(c_val)
                except:
                    continue
                if c_val == 0:
                    continue
                info = variables.get(vn)
                if info is None:
                    continue
                linear_terms.append((vn, c_val))

            if len(linear_terms) < 2:
                continue

            for idx, (target_vn, target_c) in enumerate(linear_terms):
                if target_c == 0:
                    continue
                target_info = variables.get(target_vn)
                if target_info is None:
                    continue
                cur_lb = int(target_info.get('lb', 0))
                cur_ub = int(target_info.get('ub', 0))
                if cur_lb >= cur_ub:
                    continue

                other_min = 0.0
                other_max = 0.0
                for j, (vn_j, c_j) in enumerate(linear_terms):
                    if j == idx:
                        continue
                    info_j = variables.get(vn_j)
                    if info_j is None:
                        continue
                    lb_j = int(info_j.get('lb', 0))
                    ub_j = int(info_j.get('ub', 0))
                    if c_j > 0:
                        other_min += c_j * lb_j
                        other_max += c_j * ub_j
                    else:
                        other_min += c_j * ub_j
                        other_max += c_j * lb_j

                new_lb = cur_lb
                new_ub = cur_ub

                import math as _math
                if rel in ('<=', '<'):
                    if target_c > 0:
                        bound = (rhs - other_min) / target_c
                        candidate = _math.floor(bound)
                        if rel == '<':
                            candidate -= 1
                        if candidate < new_ub:
                            new_ub = max(candidate, cur_lb)
                    elif target_c < 0:
                        bound = (rhs - other_min) / target_c
                        candidate = _math.ceil(bound)
                        if rel == '<':
                            candidate += 1
                        if candidate > new_lb:
                            new_lb = min(candidate, cur_ub)

                elif rel in ('>=', '>'):
                    if target_c > 0:
                        bound = (rhs - other_max) / target_c
                        candidate = _math.ceil(bound)
                        if rel == '>':
                            candidate += 1
                        if candidate > new_lb:
                            new_lb = min(candidate, cur_ub)
                    elif target_c < 0:
                        bound = (rhs - other_max) / target_c
                        candidate = _math.floor(bound)
                        if rel == '>':
                            candidate -= 1
                        if candidate < new_ub:
                            new_ub = max(candidate, cur_lb)

                elif rel in ('=', '=='):
                    if target_c > 0:
                        ub_cand = _math.floor((rhs - other_min) / target_c)
                        lb_cand = _math.ceil((rhs - other_max) / target_c)
                        if ub_cand < new_ub:
                            new_ub = max(ub_cand, cur_lb)
                        if lb_cand > new_lb:
                            new_lb = min(lb_cand, cur_ub)
                    elif target_c < 0:
                        lb_cand = _math.ceil((rhs - other_min) / target_c)
                        ub_cand = _math.floor((rhs - other_max) / target_c)
                        if ub_cand < new_ub:
                            new_ub = max(ub_cand, cur_lb)
                        if lb_cand > new_lb:
                            new_lb = min(lb_cand, cur_ub)

                if new_lb > new_ub:
                    continue

                if new_lb > cur_lb or new_ub < cur_ub:
                    variables[target_vn]['lb'] = new_lb
                    variables[target_vn]['ub'] = new_ub
                    changed = True
                    total_tightened += (cur_ub - cur_lb) - (new_ub - new_lb)

        if not changed:
            break

    return total_tightened


def _eval_objective(terms: List[Dict[str, Any]], sol: Dict[str, int]) -> float:
    """Evaluate objective value for a given solution."""
    total = 0.0
    for t in terms:
        c_val = t.get('c', 0)
        try:
            c_val = float(c_val) if not isinstance(c_val, (int, float)) else float(c_val)
        except:
            continue
        vs = t.get('vars', {}) or {}
        prod = 1
        for vn, p in vs.items():
            p = int(p)
            if p > 0:
                prod *= (sol.get(vn, 0) ** p)
        total += c_val * prod
    return total


def _check_constraints(constraints: List[Dict[str, Any]], sol: Dict[str, int]) -> bool:
    """Check if solution satisfies all constraints."""
    for con in constraints:
        terms = con.get('terms', con.get('lhs', []))
        rel = con.get('rel', con.get('sense', '<='))
        rhs = float(con.get('rhs', 0))
        lhs_val = 0.0
        for t in terms:
            c_val = float(t.get('c', 0))
            vs = t.get('vars', {}) or {}
            prod = 1
            for vn, p in vs.items():
                p = int(p)
                if p > 0:
                    prod *= (sol.get(vn, 0) ** p)
            lhs_val += c_val * prod
        eps = 1e-6
        if rel in ('<=', '<') and lhs_val > rhs + eps:
            return False
        if rel in ('>=', '>') and lhs_val < rhs - eps:
            return False
        if rel in ('=', '==') and abs(lhs_val - rhs) > eps:
            return False
    return True


def _greedy_initial_solution(
    terms: List[Dict[str, Any]],
    variables: Dict[str, Dict[str, Any]],
    constraints: List[Dict[str, Any]]
) -> Dict[str, int]:
    """Compute a greedy initial solution using coordinate descent over all terms
    (including quadratic), with constraint feasibility checks."""
    if not terms or not variables:
        return {}

    var_names_list = sorted(variables.keys())
    bounds = {vn: (int(info.get('lb', 0)), int(info.get('ub', 0)))
              for vn, info in variables.items()}

    # Start with lb for negative-coeff linear terms, ub for positive (simple heuristic)
    solution: Dict[str, int] = {}
    for vn in var_names_list:
        lb, ub = bounds[vn]
        solution[vn] = lb

    # Coordinate descent: for each variable, try all values in domain and pick
    # the one minimizing the objective while keeping constraints feasible.
    # For binary vars the domain is {0,1}; for general integer, sample strategically.
    max_passes = 3
    best_obj = _eval_objective(terms, solution)
    has_constraints = bool(constraints)

    for _ in range(max_passes):
        improved = False
        for vn in var_names_list:
            lb, ub = bounds[vn]
            if lb == ub:
                continue
            orig_val = solution[vn]
            best_val = orig_val
            best_local = best_obj

            if ub - lb <= 20:
                candidates = range(lb, ub + 1)
            else:
                mid = (lb + ub) // 2
                candidates = sorted(set([lb, ub, mid, (lb + mid) // 2, (mid + ub) // 2]))

            for v in candidates:
                if v == orig_val:
                    continue
                solution[vn] = v
                obj = _eval_objective(terms, solution)
                if obj < best_local:
                    if not has_constraints or _check_constraints(constraints, solution):
                        best_local = obj
                        best_val = v

            solution[vn] = best_val
            if best_val != orig_val:
                best_obj = best_local
                improved = True

        if not improved:
            break

    if has_constraints and not _check_constraints(constraints, solution):
        fallback = {vn: bounds[vn][0] for vn in var_names_list}
        if _check_constraints(constraints, fallback):
            solution = fallback

    return solution


def _objective_lower_bound_shift(terms: List[Dict[str, Any]], variables: Dict[str, Dict[str, Any]]) -> int:
    lb = 0
    for t in terms or []:
        c = t.get('c', 0)
        if isinstance(c, Fraction):
            c_val = c.numerator / (c.denominator if c.denominator != 0 else 1)
        else:
            c_val = float(c)
        if c_val >= 0:
            continue
        prod = 1
        for vn, k in (t.get('vars', {}) or {}).items():
            ub = int(variables.get(vn, {}).get('ub', 0))
            prod *= (ub ** int(k))
        lb += int(c_val * prod)
    return -lb if lb < 0 else 0


def preprocess_problem(problem: Dict[str, Any],
                      denom_cap: int = 10**6,
                      scale_limit: int = 10**6,
                      enable_min_to_max: bool = True,
                      enable_integerize: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    prob = copy.deepcopy(problem or {})
    meta: Dict[str, Any] = {
        'variable_shifts': {},
        'objective': {
            'sense_flip': False,
            'scale': 1,
            'constant_shift': 0,
            'lower_bound_shift': 0,
        },
        'constraints': []
    }

    # 1) Constraint propagation for domain tightening (before shifting)
    variables = prob.get('variables', {}) or {}
    raw_constraints = prob.get('constraints', []) or []
    if raw_constraints:
        tightened = _bound_tightening(raw_constraints, variables, max_rounds=10)
        if tightened > 0:
            print(f"  [preprocess] bound tightening reduced domain by {tightened} total units")

    # 1.1) Variable shift mapping
    shift_map: Dict[str, int] = {}
    for vn, info in variables.items():
        lb = int(info.get('lb', 0)) if info is not None else 0
        ub = int(info.get('ub', 0)) if info is not None else 0
        if lb != 0:
            shift_map[vn] = lb
            new_ub = max(0, ub - lb)
            variables[vn]['lb'] = 0
            variables[vn]['ub'] = new_ub
    meta['variable_shifts'] = shift_map

    # 2) Objective: shift-expand + merge
    obj = prob.get('objective', {}) or {}
    sense = str(obj.get('sense', 'max')).lower()
    obj_terms = obj.get('terms', []) or []
    obj_terms_new, obj_const = _canonicalize_polynomial(obj_terms, shift_map)

    # 2.05) For binary vars, reduce x^k to x
    obj_terms_new = _reduce_binary_powers(obj_terms_new, variables)

    # 2.1) min -> max (negate coefficients)
    if enable_min_to_max and sense in ('min', 'minimize', 'minimum'):
        for t in obj_terms_new:
            t['c'] = -_safe_fraction(t['c'], denom_cap)
        obj_const = -obj_const
        prob['objective']['sense'] = 'max'
        meta['objective']['sense_flip'] = True

    # 2.2) Integerize objective (higher precision for soft weights)
    obj_lcm, obj_gcd = 1, 1
    if enable_integerize:
        obj_scale_limit = scale_limit * 100
        obj_terms_new, _, (obj_lcm, obj_gcd) = _integerize_polynomial(obj_terms_new, None, denom_cap, obj_scale_limit)
        meta['objective']['scale_lcm'] = obj_lcm
        meta['objective']['scale_gcd'] = obj_gcd
        # Keep backward-compat 'scale' key (net factor, may lose precision for recovery)
        meta['objective']['scale'] = max(1, obj_lcm // obj_gcd)

    # 2.25) Simple binary probing: fix variables with provable optimal direction
    constraints_raw = prob.get('constraints', []) or []
    probed = _probe_binary_variables(obj_terms_new, variables, constraints_raw)
    if probed:
        obj_terms_new, probe_const = _substitute_fixed_variables(obj_terms_new, probed)
        if enable_integerize and obj_lcm > 0:
            obj_const += Fraction(int(round(probe_const)), 1) * Fraction(obj_gcd, obj_lcm)
        else:
            obj_const += probe_const
        # Substitute fixed variables into constraints as well
        for cons in constraints_raw:
            cons_terms = cons.get('terms', []) or []
            new_terms, added_const = _substitute_fixed_variables(cons_terms, probed)
            cons['terms'] = new_terms
            if added_const != 0:
                old_rhs = cons.get('rhs', 0)
                try:
                    old_rhs = float(old_rhs) if not isinstance(old_rhs, (int, float)) else float(old_rhs)
                except:
                    old_rhs = 0
                cons['rhs'] = old_rhs - added_const
        for vn in probed:
            if vn in variables:
                del variables[vn]
        print(f"  [preprocess] probing fixed {len(probed)} binary variables: "
              f"{dict(list(probed.items())[:5])}{'...' if len(probed) > 5 else ''}")
    meta['objective']['probed_variables'] = probed

    # 2.3) Record constant (not encoded; restored after solving)
    if obj_const != 0:
        meta['objective']['constant_shift'] = float(obj_const)
    prob['objective']['terms'] = obj_terms_new

    # 2.4) Negative-weight safe lower-bound shift (metadata only)
    meta['objective']['lower_bound_shift'] = _objective_lower_bound_shift(obj_terms_new, variables)

    # 2.5) Greedy initial solution for warm-starting
    constraints_raw_for_greedy = prob.get('constraints', []) or []
    greedy_sol = _greedy_initial_solution(obj_terms_new, variables, constraints_raw_for_greedy)
    meta['objective']['greedy_solution'] = greedy_sol

    # 3) Constraints: shift-expand + absorb constant into rhs + integerize
    new_constraints: List[Dict[str, Any]] = []
    for cons in prob.get('constraints', []) or []:
        sense = cons.get('sense', cons.get('rel', '<='))
        rhs = cons.get('rhs', 0)
        terms = cons.get('terms', []) or []
        terms_new, const_sum = _canonicalize_polynomial(terms, shift_map)
        terms_new = _reduce_binary_powers(terms_new, variables)
        # Absorb constant into rhs
        rhs_frac = _safe_fraction(rhs, denom_cap) - const_sum
        if enable_integerize:
            terms_int, rhs_int, (_lcm, _gcd) = _integerize_polynomial(terms_new, rhs_frac, denom_cap, scale_limit)
        else:
            terms_int = []
            for t in terms_new:
                cf = _safe_fraction(t['c'], denom_cap)
                terms_int.append({'c': int(cf.numerator // (cf.denominator if cf.denominator != 0 else 1)),
                                  'vars': dict(t['vars'])})
            rf = _safe_fraction(rhs_frac, denom_cap)
            rhs_int = int(rf.numerator // (rf.denominator if rf.denominator != 0 else 1))

        new_constraints.append({
            'sense': sense,
            'rhs': rhs_int,
            'terms': terms_int
        })
        meta['constraints'].append({'constant_moved': float(const_sum)})

    prob['constraints'] = new_constraints

    return prob, meta
