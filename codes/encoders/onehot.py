#!/usr/bin/env python3

import itertools
import time
from typing import Dict, Any, List, Tuple, Optional
from pysat.formula import WCNF, IDPool


class _EncodingTimeout(RuntimeError):
    pass


def encode_term_oh(
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
                raise _EncodingTimeout("OH term encoding exceeded deadline")
        weight = c
        for i, var_name in enumerate(sorted_var_names):
            k_q = int(term_vars[var_name])
            r_q = r_tuple[i]
            lb = int(lb_map.get(var_name, 0)) if lb_map else 0
            actual = r_q + lb
            weight *= (actual ** k_q)

        if weight == 0:
            continue

        # Optimization: for linear terms (single variable), use literal directly
        if len(sorted_var_names) == 1:
            q = sorted_var_indices[0]
            r_q = r_tuple[0]
            z_var = x_fn(q, r_q)
            z_mapping[r_tuple] = (z_var, weight)
            continue

        # Optimization: negative-weight objective terms -> direct clause
        if for_objective and weight < 0:
            neg_lits = []
            for i, q in enumerate(sorted_var_indices):
                r_q = r_tuple[i]
                neg_lits.append(-x_fn(q, r_q))
            direct_soft.append((neg_lits, -weight))
            continue

        z_var = vpool.id(f'{z_prefix}_{r_tuple}')
        z_mapping[r_tuple] = (z_var, weight)

        if for_objective and weight > 0:
            # Objective only: OH-H1 suffices (OH-H2 redundant for c>0, per paper §3.1)
            for i, q in enumerate(sorted_var_indices):
                r_q = r_tuple[i]
                hard_clauses.append([-z_var, x_fn(q, r_q)])
        elif for_objective and weight < 0:
            # Objective only: OH-H2 suffices (OH-H1 redundant for c<0, per paper §3.1)
            clause = [z_var]
            for i, q in enumerate(sorted_var_indices):
                r_q = r_tuple[i]
                clause.append(-x_fn(q, r_q))
            hard_clauses.append(clause)
        else:
            # Constraints: need full biconditional (OH-H1 + OH-H2)
            for i, q in enumerate(sorted_var_indices):
                r_q = r_tuple[i]
                hard_clauses.append([-z_var, x_fn(q, r_q)])
            clause = [z_var]
            for i, q in enumerate(sorted_var_indices):
                r_q = r_tuple[i]
                clause.append(-x_fn(q, r_q))
            hard_clauses.append(clause)

    return z_mapping, hard_clauses, direct_soft


def encode_polynomial_oh(
    terms: List[Dict[str, Any]],
    problem: Dict[str, Any],
    name2idx: Dict[str, int],
    vpool: IDPool,
    z_prefix: str = "z_oh",
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
        z_map, hard, direct = encode_term_oh(
            term, problem, name2idx, vpool, t_prefix, lb_map, for_objective,
            deadline=deadline
        )

        for r_tuple, (z_var, weight) in z_map.items():
            key = (t_idx,) + r_tuple
            all_z_mapping[key] = (z_var, weight)
        all_hard.extend(hard)
        all_direct.extend(direct)

    return all_z_mapping, all_hard, all_direct


def encode_objective_oh(
    problem: Dict[str, Any],
    name2idx: Dict[str, int],
    vpool: IDPool,
    lb_map: Dict[str, int] = None
) -> WCNF:
    w = WCNF()
    terms = problem['objective'].get('terms', [])

    if not terms:
        return w

    z_mapping, hard_clauses, direct_soft = encode_polynomial_oh(
        terms, problem, name2idx, vpool, "z_oh_obj", lb_map, for_objective=True
    )

    for clause in hard_clauses:
        w.append(clause)

    neg_soft_total = 0
    for _, (z_var, weight) in z_mapping.items():
        wt = int(weight)
        if wt > 0:
            w.append([z_var], weight=wt)
        elif wt < 0:
            w.append([-z_var], weight=(-wt))
            neg_soft_total += (-wt)

    # Direct soft clauses from negative-weight cross-product terms
    for clause_lits, abs_weight in direct_soft:
        w.append(clause_lits, weight=int(abs_weight))
        neg_soft_total += int(abs_weight)

    if neg_soft_total > 0:
        w._neg_soft_total = neg_soft_total

    return w
