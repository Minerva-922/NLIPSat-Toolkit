#!/usr/bin/env python3
"""Decision solving via NLIP encoding + SAT backend.

This script is intended for SMT decision benchmarks:
  SMT2 -> NLIP constraint extraction (objective=zero) -> CNF hard clauses -> SAT
"""

import argparse
import json
import os
import signal
import sys
import time

CODES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "codes")
sys.path.insert(0, CODES_DIR)

from pysat.solvers import Solver
from solver.solve import build_wcnf, EncodingTimeoutError
from solver.config import EncodingConfig
from encoders.onehot import _EncodingTimeout


def _timeout_handler(name, encoding, sat_solver, start_time):
    elapsed = time.time() - start_time
    print(
        f">>> Benchmark {name} "
        f"Encoding {encoding} "
        f"Solver {sat_solver} "
        f"Status TIMEOUT "
        f"OPT N/A "
        f"TimeRead 0.000 TimeEncode 0.000 TimeSolve 0.000 TimeTotal {elapsed:.3f}"
    )
    sys.stdout.flush()
    os._exit(0)


def load_problem_for_decision(input_path):
    if input_path.endswith(".smt2"):
        from tools.smt2_parser import parse_smt2_file
        # Decision mode: no optimization objective for SMT benchmarks.
        return parse_smt2_file(input_path, objective_mode="zero")
    if input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if input_path.endswith(".qplib"):
        from tools.qplib_parser import parse_qplib_file
        return parse_qplib_file(input_path)
    raise ValueError(f"Unsupported input format: {input_path}")


def sat_backend_name(name):
    name = name.upper()
    if name == "CADICAL":
        return "cadical195"
    if name == "GLUCOSE":
        return "glucose4"
    raise ValueError(f"Unsupported SAT backend: {name}")


def main():
    parser = argparse.ArgumentParser(description="NLIP-encoded SAT baseline")
    parser.add_argument("input", help="Input file (SMT2 preferred)")
    parser.add_argument("-e", "--encoding", choices=["OH", "UNA", "BIN"], default="OH")
    parser.add_argument("--decomp", action="store_true", help="Enable decomposition (BIN only)")
    parser.add_argument("--decomp-threshold", type=int, default=3)
    parser.add_argument("--decomp-strategy", choices=["sequential", "binary_tree"], default="sequential")
    parser.add_argument("--sat-solver", choices=["CADICAL", "GLUCOSE"], required=True)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--no-preprocess", action="store_true")
    args = parser.parse_args()

    problem_name = os.path.basename(args.input)
    t_start = time.time()

    if args.timeout and hasattr(signal, "SIGALRM"):
        signal.signal(
            signal.SIGALRM,
            lambda signum, frame: _timeout_handler(problem_name, args.encoding, args.sat_solver, t_start),
        )
        signal.alarm(args.timeout)

    t_read0 = time.time()
    try:
        problem = load_problem_for_decision(args.input)
    except Exception as e:
        elapsed = time.time() - t_start
        print(
            f">>> Benchmark {problem_name} "
            f"Encoding {args.encoding} Solver {args.sat_solver} "
            f"Status ERROR OPT N/A "
            f"TimeRead 0.000 TimeEncode 0.000 TimeSolve 0.000 TimeTotal {elapsed:.3f}"
        )
        print(f"Error: failed to parse input - {e}", file=sys.stderr)
        return 1
    time_read = time.time() - t_read0

    cfg = EncodingConfig()
    cfg.enable_preprocess = not args.no_preprocess
    cfg.use_decomposition = args.decomp and args.encoding == "BIN"
    cfg.decomp_threshold = args.decomp_threshold
    cfg.decomp_strategy = args.decomp_strategy
    cfg.decomp_exact = True
    cfg.decomp_shared = True
    cfg.weight_gcd_normalize = True
    cfg.external_solver_timeout = 0

    try:
        wcnf, _, timings, _ = build_wcnf(problem, args.encoding, cfg, collect_timings=True)
    except (EncodingTimeoutError, _EncodingTimeout):
        elapsed = time.time() - t_start
        print(
            f">>> Benchmark {problem_name} "
            f"Encoding {args.encoding} Solver {args.sat_solver} "
            f"Status TIMEOUT OPT N/A "
            f"TimeRead {time_read:.3f} TimeEncode 0.000 TimeSolve 0.000 TimeTotal {elapsed:.3f}"
        )
        return 0
    except Exception as e:
        elapsed = time.time() - t_start
        print(
            f">>> Benchmark {problem_name} "
            f"Encoding {args.encoding} Solver {args.sat_solver} "
            f"Status ERROR OPT N/A "
            f"TimeRead {time_read:.3f} TimeEncode 0.000 TimeSolve 0.000 TimeTotal {elapsed:.3f}"
        )
        print(f"Error: encoding failed - {e}", file=sys.stderr)
        return 1

    # For SMT decision mode we expect no soft clauses; if present, treat as unsupported.
    if len(wcnf.soft) > 0:
        elapsed = time.time() - t_start
        print(
            f">>> Benchmark {problem_name} "
            f"Encoding {args.encoding} Solver {args.sat_solver} "
            f"Status UNSUPPORTED OPT N/A "
            f"TimeRead {time_read:.3f} TimeEncode {timings.get('build_total', 0.0):.3f} "
            f"TimeSolve 0.000 TimeTotal {elapsed:.3f}"
        )
        print("Error: decision SAT mode requires zero soft clauses.", file=sys.stderr)
        return 1

    t_solve0 = time.time()
    backend = sat_backend_name(args.sat_solver)
    status = "UNKNOWN"
    try:
        with Solver(name=backend, bootstrap_with=wcnf.hard) as sat_solver:
            sat_res = sat_solver.solve()
        status = "SAT" if sat_res else "UNSAT"
    except Exception as e:
        status = "ERROR"
        print(f"Error: SAT backend failed ({backend}) - {e}", file=sys.stderr)
    time_solve = time.time() - t_solve0

    elapsed = time.time() - t_start
    enc_str = args.encoding + ("+DECOMP" if cfg.use_decomposition else "")
    print(
        f">>> Benchmark {problem_name} "
        f"Encoding {enc_str} "
        f"Solver {args.sat_solver} "
        f"Status {status} "
        f"OPT N/A "
        f"Vars {wcnf.nv} Hard {len(wcnf.hard)} Soft {len(wcnf.soft)} TopW {wcnf.topw} "
        f"TimeRead {time_read:.3f} "
        f"TimeEncode {timings.get('build_total', 0.0):.3f} "
        f"TimeSolve {time_solve:.3f} "
        f"TimeTotal {elapsed:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
