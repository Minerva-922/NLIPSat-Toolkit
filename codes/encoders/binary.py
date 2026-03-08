#!/usr/bin/env python3

import math
import itertools
from typing import Dict, Any, List, Tuple
from pysat.formula import WCNF, IDPool


def encode_term_bin(
    term: Dict[str, Any],
    problem: Dict[str, Any],
    name2idx: Dict[str, int],
    vpool: IDPool,
    z_prefix: str,
    lb_map: Dict[str, int] = None,
    for_objective: bool = False,
    and_cache: Dict = None
) -> Tuple[Dict[Tuple, Tuple[int, int]], List[List[int]], List[Tuple[List[int], int]]]:

    term_vars = term.get('vars', {}) or {}
    c = int(term.get('c', 1))

    if not term_vars or c == 0:
        return {}, [], []

    # Build Omega: expand term into k factor slots
    omega = []
    for var_name, power in sorted(term_vars.items()):
        q = name2idx[var_name]
        ub = int(problem['variables'][var_name]['ub'])
        for _ in range(int(power)):
            omega.append((var_name, q, ub))

    k = len(omega)
    if k == 0:
        return {}, [], []

    # FIX: per-variable bit count (was global max, causing phantom bits for mixed ub)
    bit_ranges = []
    for _, _, ub in omega:
        nb = math.ceil(math.log2(ub + 1)) if ub > 0 else 1
        bit_ranges.append(range(nb))

    x_fn = lambda q, r: vpool.id(f'x_{q}@{r}')

    z_mapping = {}
    hard_clauses = []
    direct_soft = []

    for r_tuple in itertools.product(*bit_ranges):
        weight = c * (2 ** sum(r_tuple))

        # Optimization: for linear terms (k=1), use the variable literal directly
        if k == 1:
            _, q, _ = omega[0]
            r_i = r_tuple[0]
            z_var = x_fn(q, r_i)
            z_mapping[r_tuple] = (z_var, weight)
            continue

        # Optimization: negative-weight objective terms -> direct k-literal soft clause
        # z <-> AND(x_i) with soft [-z] w  ===  soft [¬x_1, ..., ¬x_k] w
        if for_objective and weight < 0:
            neg_lits = []
            seen = set()
            for i in range(k):
                _, q, _ = omega[i]
                r_i = r_tuple[i]
                lit = -x_fn(q, r_i)
                if lit not in seen:
                    neg_lits.append(lit)
                    seen.add(lit)
            direct_soft.append((neg_lits, -weight))
            continue

        # Build the set of input literals for AND gate
        and_inputs = []
        seen_inputs = set()
        for i in range(k):
            _, q, _ = omega[i]
            r_i = r_tuple[i]
            x_lit = x_fn(q, r_i)
            if x_lit not in seen_inputs:
                and_inputs.append(x_lit)
                seen_inputs.add(x_lit)

        and_key = frozenset(and_inputs) if and_cache is not None else None
        cached = False
        if and_cache is not None and and_key in and_cache:
            z_var = and_cache[and_key]
            cached = True
        else:
            z_var = vpool.id(f'{z_prefix}_{r_tuple}')
            if and_cache is not None:
                and_cache[and_key] = z_var

        z_mapping[r_tuple] = (z_var, weight)

        if not cached:
            for x_lit in and_inputs:
                hard_clauses.append([-z_var, x_lit])
            if not (for_objective and weight > 0):
                clause = [z_var] + [-x_lit for x_lit in and_inputs]
                hard_clauses.append(clause)

    return z_mapping, hard_clauses, direct_soft


def encode_polynomial_bin(
    terms: List[Dict[str, Any]],
    problem: Dict[str, Any],
    name2idx: Dict[str, int],
    vpool: IDPool,
    z_prefix: str = "z_bin",
    lb_map: Dict[str, int] = None,
    for_objective: bool = False,
    external_and_cache: Dict = None
) -> Tuple[Dict[Tuple, Tuple[int, int]], List[List[int]], List[Tuple[List[int], int]]]:
    if not terms:
        return {}, [], []

    all_z_mapping = {}
    all_hard = []
    all_direct = []

    and_cache = external_and_cache if external_and_cache is not None else {}

    for t_idx, term in enumerate(terms):
        t_prefix = f"{z_prefix}_t{t_idx}"
        z_map, hard, direct = encode_term_bin(
            term, problem, name2idx, vpool, t_prefix, lb_map, for_objective,
            and_cache=and_cache
        )

        for r_tuple, (z_var, weight) in z_map.items():
            key = (t_idx,) + r_tuple
            all_z_mapping[key] = (z_var, weight)
        all_hard.extend(hard)
        all_direct.extend(direct)

    return all_z_mapping, all_hard, all_direct


def encode_objective_bin(
    problem: Dict[str, Any],
    name2idx: Dict[str, int],
    vpool: IDPool,
    lb_map: Dict[str, int] = None,
    external_and_cache: Dict = None
) -> WCNF:
    w = WCNF()
    terms = problem['objective'].get('terms', [])

    if not terms:
        return w

    z_mapping, hard_clauses, direct_soft = encode_polynomial_bin(
        terms, problem, name2idx, vpool, "z_bin_obj", lb_map, for_objective=True,
        external_and_cache=external_and_cache
    )

    for clause in hard_clauses:
        w.append(clause)

    neg_soft_total = 0
    for _, (z_var, weight) in z_mapping.items():
        wt = int(weight)
        if wt > 0:
            w.append([z_var], weight=wt)
        elif wt < 0:
            # Linear terms with negative weight still go through z_mapping
            w.append([-z_var], weight=(-wt))
            neg_soft_total += (-wt)

    # Direct soft clauses from negative-weight cross-product terms
    for clause_lits, abs_weight in direct_soft:
        w.append(clause_lits, weight=int(abs_weight))
        neg_soft_total += int(abs_weight)

    if neg_soft_total > 0:
        w._neg_soft_total = neg_soft_total

    return w


def encode_objective_bin_with_decomposition(
    problem: Dict[str, Any],
    name2idx: Dict[str, int],
    vpool: IDPool,
    lb_map: Dict[str, int] = None,
    decomp_threshold: int = 3,
    decomp_strategy: str = "sequential",
    decomp_exact: bool = True,
    decomp_shared: bool = True
) -> WCNF:
    from .decomposition import encode_term_decomp, MulCache

    w = WCNF()
    terms = problem['objective'].get('terms', [])

    if not terms:
        return w

    # Shared substructure cache (paper Figure 3c)
    cache = MulCache() if decomp_shared else None

    neg_soft_total = 0

    for t_idx, term in enumerate(terms):
        tv = term.get('vars', {}) or {}
        degree = sum(int(p) for p in tv.values())
        c = int(term.get('c', 1))

        # Determine sense: exact (paper default) or relaxed
        if decomp_exact:
            sense = 0
        else:
            sense = 1 if c > 0 else -1

        # Adaptive decomposition: also trigger for quadratic terms with large
        # bit-product (many AND gates in standard BIN encoding).
        use_decomp_for_term = degree >= decomp_threshold
        if not use_decomp_for_term and degree >= 2 and len(tv) >= 2:
            bit_product = 1
            for vn, p in tv.items():
                ub = int(problem['variables'][vn]['ub'])
                nb = math.ceil(math.log2(ub + 1)) if ub > 0 else 1
                bit_product *= nb ** int(p)
            if bit_product > 256:
                use_decomp_for_term = True

        if use_decomp_for_term:
            z_mapping, hard_clauses = encode_term_decomp(
                term, problem, name2idx, vpool, f"decomp_t{t_idx}", lb_map,
                sense=sense, strategy=decomp_strategy, cache=cache
            )
            direct_soft = []
        else:
            z_mapping, hard_clauses, direct_soft = encode_term_bin(
                term, problem, name2idx, vpool, f"z_bin_t{t_idx}", lb_map,
                for_objective=True
            )

        for clause in hard_clauses:
            w.append(clause)

        for _, (z_var, weight) in z_mapping.items():
            wt = int(weight)
            if wt > 0:
                w.append([z_var], weight=wt)
            elif wt < 0:
                w.append([-z_var], weight=(-wt))
                neg_soft_total += (-wt)

        for clause_lits, abs_weight in direct_soft:
            w.append(clause_lits, weight=int(abs_weight))
            neg_soft_total += int(abs_weight)

    if neg_soft_total > 0:
        w._neg_soft_total = neg_soft_total

    if cache is not None and len(cache) > 0:
        print(f"  [decomp] shared Mul cache: {len(cache)} products reused")

    return w
