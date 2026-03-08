#!/usr/bin/env python3

from pysat.formula import WCNF, IDPool
from pysat.card import CardEnc
import math


class EncodingDriverError(Exception):
    pass


def add_domain_constraints(problem, encoding, name2idx, vpool, wcnf, cfg=None):
    x_fn = lambda q, r: vpool.id(f'x_{q}@{r}')
    use_map = cfg is not None and getattr(cfg, 'use_mapping_shift', False)

    for var_name, meta in problem.get('variables', {}).items():
        q = name2idx[var_name]
        lb = int(meta.get('lb', 0))
        ub = int(meta.get('ub', 0))
        ub_eff = (ub - lb) if use_map else ub

        if encoding == 'OH':
            _ensure_oh_structure(wcnf, x_fn, q, ub_eff, vpool)
        elif encoding == 'UNA':
            _ensure_una_structure(wcnf, x_fn, q, ub_eff)
        elif encoding == 'BIN':
            _ensure_bin_structure(wcnf, x_fn, q, ub_eff)
        else:
            raise EncodingDriverError(f"Unsupported encoding: {encoding}")


def encode_constraints(problem, encoding, name2idx, vpool, cfg=None):
    from .constraints import encode_all_constraints

    use_map = cfg is not None and getattr(cfg, 'use_mapping_shift', False)
    lb_map = {vn: int(info.get('lb', 0))
              for vn, info in problem.get('variables', {}).items()} if use_map else {}

    global_cache = getattr(cfg, '_global_and_cache', None) if cfg else None
    deadline = getattr(cfg, 'encoding_deadline', 0.0) if cfg else 0.0

    use_decomp = (
        encoding == 'BIN' and
        cfg is not None and
        getattr(cfg, 'use_decomposition', False)
    )
    decomp_threshold = getattr(cfg, 'decomp_threshold', 3) if cfg else 3
    decomp_strategy = getattr(cfg, 'decomp_strategy', 'sequential') if cfg else 'sequential'
    decomp_shared = getattr(cfg, 'decomp_shared', True) if cfg else True

    clauses = encode_all_constraints(
        problem, name2idx, vpool, encoding, lb_map,
        and_cache=global_cache, deadline=deadline,
        use_decomp=use_decomp,
        decomp_threshold=decomp_threshold,
        decomp_strategy=decomp_strategy,
        decomp_shared=decomp_shared
    )

    w = WCNF()
    for clause in clauses:
        w.append(clause)

    return w


def _is_all_binary(problem):
    """Check whether every variable has domain {0,1}."""
    for info in problem.get('variables', {}).values():
        if int(info.get('lb', 0)) != 0 or int(info.get('ub', 0)) != 1:
            return False
    return True


def _encode_objective_binary_linear(problem, name2idx, vpool, encoding='BIN'):
    """Fast McCormick linearization for objectives over binary variables.
    Each quadratic term x_i*x_j is linearized via z <-> (x_i AND x_j).
    Works for BIN (x_q@0 = value), OH/UNA (x_q@1 = indicator for value 1)."""
    from pysat.formula import WCNF
    w = WCNF()
    terms = problem.get('objective', {}).get('terms', [])
    if not terms:
        return w

    bit_idx = 0 if encoding == 'BIN' else 1
    x_fn = lambda q: vpool.id(f'x_{q}@{bit_idx}')
    product_cache = {}
    neg_soft_total = 0

    for t in terms:
        c_val = int(t.get('c', 0))
        if c_val == 0:
            continue
        vs = t.get('vars', {}) or {}
        active = [(vn, int(p)) for vn, p in vs.items() if int(p) > 0]

        if not active:
            continue

        if len(active) == 1:
            vn, _ = active[0]
            lit = x_fn(name2idx[vn])
            if c_val > 0:
                w.append([lit], weight=c_val)
            else:
                w.append([-lit], weight=-c_val)
                neg_soft_total += -c_val
            continue

        if len(active) == 2:
            (vn1, _), (vn2, _) = active
            x1, x2 = x_fn(name2idx[vn1]), x_fn(name2idx[vn2])
            cache_key = (min(x1, x2), max(x1, x2))
            if cache_key in product_cache:
                z = product_cache[cache_key]
            else:
                z = vpool.id(f'z_binlin_{name2idx[vn1]}_{name2idx[vn2]}')
                w.append([-z, x1])
                w.append([-z, x2])
                w.append([z, -x1, -x2])
                product_cache[cache_key] = z
            if c_val > 0:
                w.append([z], weight=c_val)
            else:
                w.append([-z], weight=-c_val)
                neg_soft_total += -c_val
            continue

        # Higher-degree: chain AND gates (x^k = x for binary, so degree <= #vars)
        var_lits = []
        for vn, _ in active:
            lit = x_fn(name2idx[vn])
            if lit not in var_lits:
                var_lits.append(lit)
        z = var_lits[0]
        for i in range(1, len(var_lits)):
            pair_key = (min(z, var_lits[i]), max(z, var_lits[i]))
            if pair_key in product_cache:
                z = product_cache[pair_key]
            else:
                z_new = vpool.id(f'z_binlin_chain_{pair_key[0]}_{pair_key[1]}')
                w.append([-z_new, z])
                w.append([-z_new, var_lits[i]])
                w.append([z_new, -z, -var_lits[i]])
                product_cache[pair_key] = z_new
                z = z_new
        if c_val > 0:
            w.append([z], weight=c_val)
        else:
            w.append([-z], weight=-c_val)
            neg_soft_total += -c_val

    if neg_soft_total > 0:
        w._neg_soft_total = neg_soft_total
    return w


def encode_objective(problem, encoding, name2idx, vpool, cfg=None):
    use_map = cfg is not None and getattr(cfg, 'use_mapping_shift', False)
    lb_map = {vn: int(info.get('lb', 0))
              for vn, info in problem.get('variables', {}).items()} if use_map else {}

    if _is_all_binary(problem):
        return _encode_objective_binary_linear(problem, name2idx, vpool, encoding=encoding)

    # Auto-downgrade OH/UNA to BIN when combinatorial space is too large
    effective_encoding = encoding
    if encoding in ('OH', 'UNA'):
        variables = problem.get('variables', {})
        terms = problem.get('objective', {}).get('terms', [])
        max_space = 0
        for t in terms:
            tv = t.get('vars', {}) or {}
            if len(tv) < 2:
                continue
            space = 1
            for vn in tv:
                ub = int(variables.get(vn, {}).get('ub', 0))
                space *= (ub + 1)
                if space > 1_000_000:
                    break
            max_space = max(max_space, space)
        if max_space > 1_000_000:
            effective_encoding = 'BIN'
            print(f"  [auto-downgrade] {encoding}→BIN: max term space {max_space} > 1M")

    # Check if decomposition should be used (BIN only)
    use_decomp = (
        effective_encoding == 'BIN' and
        cfg is not None and
        getattr(cfg, 'use_decomposition', False)
    )
    decomp_threshold = getattr(cfg, 'decomp_threshold', 3) if cfg else 3

    if effective_encoding == 'OH':
        from .onehot import encode_objective_oh
        return encode_objective_oh(problem, name2idx, vpool, lb_map=lb_map)

    elif effective_encoding == 'UNA':
        from .unary import encode_objective_una
        return encode_objective_una(problem, name2idx, vpool, lb_map=lb_map)

    elif effective_encoding == 'BIN':
        if use_decomp:
            from .binary import encode_objective_bin_with_decomposition
            return encode_objective_bin_with_decomposition(
                problem, name2idx, vpool,
                lb_map=lb_map,
                decomp_threshold=decomp_threshold,
                decomp_strategy=getattr(cfg, 'decomp_strategy', 'sequential'),
                decomp_exact=getattr(cfg, 'decomp_exact', True),
                decomp_shared=getattr(cfg, 'decomp_shared', True)
            )
        else:
            from .binary import encode_objective_bin
            global_cache = getattr(cfg, '_global_and_cache', None) if cfg else None
            return encode_objective_bin(problem, name2idx, vpool, lb_map=lb_map,
                                        external_and_cache=global_cache)

    raise EncodingDriverError(f"Unsupported encoding: {encoding}")


def _ensure_oh_structure(w, x_fn, q, ub, vpool):
    lits = [x_fn(q, r) for r in range(ub + 1)]
    for clause in CardEnc.equals(lits=lits, bound=1, vpool=vpool):
        w.append(clause)


def _ensure_una_structure(w, x_fn, q, ub):
    w.append([x_fn(q, 0)])
    for r in range(1, ub + 1):
        w.append([-x_fn(q, r), x_fn(q, r - 1)])


def _ensure_bin_structure(w, x_fn, q, ub):
    num_bits = math.ceil(math.log2(ub + 1)) if ub > 0 else 1

    for val in range(ub + 1, 1 << num_bits):
        clause = []
        for r in range(num_bits):
            if (val >> r) & 1:
                clause.append(-x_fn(q, r))
            else:
                clause.append(x_fn(q, r))
        w.append(clause)


