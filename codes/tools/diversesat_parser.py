#!/usr/bin/env python3

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, Iterator, List


def _iter_dimacs_ints(lines: Iterator[str]) -> Iterator[int]:
    for line in lines:
        line = line.strip()
        if not line or line.startswith('c'):
            continue
        if line.startswith('p'):
            continue
        for tok in line.split():
            try:
                yield int(tok)
            except Exception:
                continue


def _parse_cnf_raw(filepath: str):
    n_vars = None
    clauses: List[List[int]] = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()

    for line in raw_lines:
        s = line.strip()
        if s.startswith('p'):
            parts = s.split()
            if len(parts) >= 4 and parts[1].lower() == 'cnf':
                try:
                    n_vars = int(parts[2])
                except Exception:
                    n_vars = None
            break

    cur: List[int] = []
    for v in _iter_dimacs_ints(iter(raw_lines)):
        if v == 0:
            clauses.append(cur)
            cur = []
        else:
            cur.append(v)
    if cur:
        clauses.append(cur)

    if n_vars is None:
        n_vars = max((abs(lit) for c in clauses for lit in c), default=0)
    if n_vars <= 0:
        n_vars = 1

    return n_vars, clauses


def parse_diversesat(filepath: str, k: int) -> Dict[str, Any]:
    if k < 2:
        raise ValueError(f"k must be >= 2 for Diverse-SAT, got {k}")

    n_vars, clauses = _parse_cnf_raw(filepath)

    # Variable naming: x{i}_m{j}  (i: original var index, j: model index)
    def vname(i: int, m: int) -> str:
        return f'x{i}_m{m}'

    # --- Variables -----------------------------------------------------------
    variables: Dict[str, Dict[str, Any]] = {}
    for i in range(1, n_vars + 1):
        for m in range(1, k + 1):
            variables[vname(i, m)] = {'lb': 0, 'ub': 1}

    # --- Constraints (each clause replicated for each model) -----------------
    constraints: List[Dict[str, Any]] = []
    for clause in clauses:
        if not clause:
            for m in range(1, k + 1):
                constraints.append({
                    'terms': [{'c': 1, 'vars': {vname(1, m): 1}}],
                    'rel': '>=',
                    'rhs': 2,
                })
            continue

        pos = set()
        neg = set()
        for lit in clause:
            if lit > 0:
                pos.add(lit)
            elif lit < 0:
                neg.add(-lit)

        if pos & neg:
            continue

        for m in range(1, k + 1):
            terms: List[Dict[str, Any]] = []
            for i in sorted(pos):
                terms.append({'c': 1, 'vars': {vname(i, m): 1}})
            for i in sorted(neg):
                terms.append({'c': -1, 'vars': {vname(i, m): 1}})
            rhs = 1 - len(neg)
            constraints.append({'terms': terms, 'rel': '>=', 'rhs': rhs})

    # --- Objective (maximize total pairwise Hamming distance) ----------------
    #
    # For each original variable i, for each model pair (a, b) with a < b:
    #   XOR(x_{i,a}, x_{i,b}) = x_{i,a} + x_{i,b} - 2 * x_{i,a} * x_{i,b}
    #
    # We aggregate coefficients to reduce term count:
    #   linear coeff of x_{i,m}  =  (k - 1)    (it appears in k-1 pairs)
    #   quadratic coeff of x_{i,a} * x_{i,b}  =  -2
    obj_terms: List[Dict[str, Any]] = []

    for i in range(1, n_vars + 1):
        for m in range(1, k + 1):
            obj_terms.append({
                'c': k - 1,
                'vars': {vname(i, m): 1}
            })

        for a, b in combinations(range(1, k + 1), 2):
            obj_terms.append({
                'c': -2,
                'vars': {vname(i, a): 1, vname(i, b): 1}
            })

    problem: Dict[str, Any] = {
        'variables': variables,
        'objective': {'sense': 'max', 'terms': obj_terms},
        'constraints': constraints,
    }

    n_pairs = k * (k - 1) // 2
    print(f"  [diversesat] n={n_vars}, k={k}, models_vars={n_vars * k}, "
          f"clauses_replicated={len(constraints)}, "
          f"obj_linear={n_vars * k}, obj_quadratic={n_vars * n_pairs}")

    return problem
