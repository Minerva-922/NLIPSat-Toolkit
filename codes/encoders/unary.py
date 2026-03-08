#!/usr/bin/env python3
import itertools
import time
from typing import Dict, Any, List, Tuple, Optional
from pysat.formula import WCNF, IDPool

from .onehot import _EncodingTimeout


def _compute_difference_def8(
    term: Dict[str, Any],
    r_tuple: Tuple[int, ...],
    sorted_var_names: List[str],
    lb_map: Dict[str, int] = None
) -> int:

    m = len(r_tuple)

    if any(r == 0 for r in r_tuple):
        return 0

    diff = 0
    for mask in range(1 << m):
        r_prime = list(r_tuple)
        popcnt = 0
        for j in range(m):
            if (mask >> j) & 1:
                r_prime[j] -= 1
                popcnt += 1

        # Evaluate term at r_prime assignment
        c = int(term.get('c', 1))
        term_vars = term.get('vars', {}) or {}
        t_val = c
        for vi, vn in enumerate(sorted_var_names):
            k_q = int(term_vars.get(vn, 0))
            if k_q == 0:
                continue
            lb = int(lb_map.get(vn, 0)) if lb_map else 0
            t_val *= ((r_prime[vi] + lb) ** k_q)

        if popcnt % 2 == 1:
            diff -= t_val
        else:
            diff += t_val

    return diff


def encode_term_una(
    term: Dict[str, Any],
    problem: Dict[str, Any],
    name2idx: Dict[str, int],
    vpool: IDPool,
    z_prefix: str,
    lb_map: Dict[str, int] = None,
    for_objective: bool = False,
    deadline: float = 0.0
) -> Tuple[Dict[Tuple, Tuple[int, int]], List[List[int]], List[Tuple[List[int], int]]]:
    term_vars = term.get('vars', {}) or {}
    c = int(term.get('c', 1))

    if not term_vars or c == 0:
        return {}, [], []

    sorted_var_names = sorted(term_vars.keys())
    sorted_var_indices = [name2idx[name] for name in sorted_var_names]

    var_ubs = []
    for name in sorted_var_names:
        info = problem['variables'][name]
        lb = int(info.get('lb', 0))
        ub = int(info.get('ub', 0))
        eff = (ub - lb) if (lb_map is not None and name in lb_map) else ub
        var_ubs.append(eff)

    x_fn = lambda q, r: vpool.id(f'x_{q}@{r}')

    z_mapping = {}
    hard_clauses = []
    direct_soft = []

    value_ranges = [range(ub + 1) for ub in var_ubs]

    _check_every = 2000
    _iter_count = 0
    for r_tuple in itertools.product(*value_ranges):
        if deadline > 0:
            _iter_count += 1
            if _iter_count % _check_every == 0 and time.time() > deadline:
                raise _EncodingTimeout("UNA term encoding exceeded deadline")
        diff_value = _compute_difference_def8(term, r_tuple, sorted_var_names, lb_map)

        if diff_value == 0:
            continue

        # Optimization: for linear terms (single variable), use literal directly
        if len(sorted_var_names) == 1:
            q = sorted_var_indices[0]
            r_q = r_tuple[0]
            z_var = x_fn(q, r_q)
            z_mapping[r_tuple] = (z_var, diff_value)
            continue

        # Optimization: negative-weight objective terms -> direct clause
        if for_objective and diff_value < 0:
            neg_lits = []
            for i, q in enumerate(sorted_var_indices):
                r_q = r_tuple[i]
                neg_lits.append(-x_fn(q, r_q))
            direct_soft.append((neg_lits, -diff_value))
            continue

        z_var = vpool.id(f'{z_prefix}_{r_tuple}')
        z_mapping[r_tuple] = (z_var, diff_value)

        if for_objective and diff_value > 0:
            # Objective only: UNA-H1 suffices (UNA-H2 redundant for positive weight)
            for i, q in enumerate(sorted_var_indices):
                r_q = r_tuple[i]
                hard_clauses.append([-z_var, x_fn(q, r_q)])
        elif for_objective and diff_value < 0:
            # Objective only: UNA-H2 suffices (UNA-H1 redundant for negative weight)
            clause = [z_var]
            for i, q in enumerate(sorted_var_indices):
                r_q = r_tuple[i]
                clause.append(-x_fn(q, r_q))
            hard_clauses.append(clause)
        else:
            # Constraints: need full biconditional (UNA-H1 + UNA-H2)
            for i, q in enumerate(sorted_var_indices):
                r_q = r_tuple[i]
                hard_clauses.append([-z_var, x_fn(q, r_q)])
            clause = [z_var]
            for i, q in enumerate(sorted_var_indices):
                r_q = r_tuple[i]
                clause.append(-x_fn(q, r_q))
            hard_clauses.append(clause)

    return z_mapping, hard_clauses, direct_soft


def encode_polynomial_una(
    terms: List[Dict[str, Any]],
    problem: Dict[str, Any],
    name2idx: Dict[str, int],
    vpool: IDPool,
    z_prefix: str = "z_una",
    lb_map: Dict[str, int] = None,
    for_objective: bool = False,
    deadline: float = 0.0
) -> Tuple[Dict[Tuple, Tuple[int, int]], List[List[int]], List[Tuple[List[int], int]]]:
    if not terms:
        return {}, [], []

    all_z_mapping = {}
    all_hard = []
    all_direct = []

    for t_idx, term in enumerate(terms):
        t_prefix = f"{z_prefix}_t{t_idx}"
        z_map, hard, direct = encode_term_una(
            term, problem, name2idx, vpool, t_prefix, lb_map, for_objective,
            deadline=deadline
        )

        for r_tuple, (z_var, diff_value) in z_map.items():
            key = (t_idx,) + r_tuple
            all_z_mapping[key] = (z_var, diff_value)
        all_hard.extend(hard)
        all_direct.extend(direct)

    return all_z_mapping, all_hard, all_direct


def encode_objective_una(
    problem: Dict[str, Any],
    name2idx: Dict[str, int],
    vpool: IDPool,
    lb_map: Dict[str, int] = None
) -> WCNF:
    w = WCNF()
    terms = problem['objective'].get('terms', [])

    if not terms:
        return w

    z_mapping, hard_clauses, direct_soft = encode_polynomial_una(
        terms, problem, name2idx, vpool, "z_una_obj", lb_map, for_objective=True
    )

    for clause in hard_clauses:
        w.append(clause)

    neg_soft_total = 0
    for _, (z_var, diff_value) in z_mapping.items():
        dv = int(diff_value)
        if dv > 0:
            w.append([z_var], weight=dv)
        elif dv < 0:
            w.append([-z_var], weight=(-dv))
            neg_soft_total += (-dv)

    # Direct soft clauses from negative-weight cross-product terms
    for clause_lits, abs_weight in direct_soft:
        w.append(clause_lits, weight=int(abs_weight))
        neg_soft_total += int(abs_weight)

    if neg_soft_total > 0:
        w._neg_soft_total = neg_soft_total

    return w
