#!/usr/bin/env python3
from __future__ import annotations

from typing import Dict, Any, List, Tuple, Iterator


def _iter_dimacs_ints(lines: Iterator[str]) -> Iterator[int]:
    for line in lines:
        line = line.strip()
        if not line or line.startswith('c'):
            continue
        if line.startswith('p'):
            # header handled elsewhere
            continue
        for tok in line.split():
            try:
                yield int(tok)
            except Exception:
                continue


def parse_cnf_file(filepath: str) -> Dict[str, Any]:
    n_vars = None
    clauses: List[List[int]] = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()

    for line in raw_lines:
        s = line.strip()
        if s.startswith('p'):
            parts = s.split()
            # p cnf <nvars> <nclauses>
            if len(parts) >= 4 and parts[1].lower() == 'cnf':
                try:
                    n_vars = int(parts[2])
                except Exception:
                    n_vars = None
            break

    ints = _iter_dimacs_ints(iter(raw_lines))
    cur: List[int] = []
    for v in ints:
        if v == 0:
            clauses.append(cur)
            cur = []
        else:
            cur.append(v)
    if cur:
        # tolerate missing trailing 0
        clauses.append(cur)

    if n_vars is None:
        # derive from max literal
        n_vars = 0
        for c in clauses:
            for lit in c:
                n_vars = max(n_vars, abs(int(lit)))

    # Guard: empty CNF is trivially SAT
    if n_vars <= 0:
        n_vars = 1

    variables: Dict[str, Dict[str, Any]] = {
        f'x{i}': {'lb': 0, 'ub': 1} for i in range(1, n_vars + 1)
    }

    constraints: List[Dict[str, Any]] = []
    for c in clauses:
        if not c:
            # Empty clause => UNSAT. Encode via an impossible constraint on x1 (binary).
            constraints.append({
                'terms': [{'c': 1, 'vars': {'x1': 1}}],
                'rel': '>=',
                'rhs': 2,
            })
            continue

        pos = set()
        neg = set()
        for lit in c:
            lit = int(lit)
            if lit > 0:
                pos.add(lit)
            elif lit < 0:
                neg.add(-lit)

        # Clause is a tautology if a var appears both positively and negatively.
        if pos & neg:
            continue

        terms: List[Dict[str, Any]] = []
        for i in sorted(pos):
            terms.append({'c': 1, 'vars': {f'x{i}': 1}})
        for i in sorted(neg):
            terms.append({'c': -1, 'vars': {f'x{i}': 1}})

        rhs = 1 - len(neg)
        constraints.append({'terms': terms, 'rel': '>=', 'rhs': rhs})

    problem: Dict[str, Any] = {
        'variables': variables,
        'objective': {'sense': 'min', 'terms': []},
        'constraints': constraints,
    }
    return problem

