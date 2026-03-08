import argparse
import json
import signal
import time
import sys
import os

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solver.solve import build_and_solve, build_wcnf, EncodingTimeoutError
from encoders.onehot import _EncodingTimeout
from solver.config import EncodingConfig
from tools.verify import verify_solution, format_verification_report

_timeout_ctx = {"problem_name": "", "encoding": "", "solver": "", "t_start": 0}

def load_json_problem(input_path):
    # Fast diagnostics for common dataset issues before parsing JSON.
    with open(input_path, 'r', encoding='utf-8') as f:
        prefix = f.read(1024)
        if not prefix.strip():
            raise ValueError("input file is empty")
        f.seek(0)
        return json.load(f)

def load_problem(input_path, k=None):
    if input_path.endswith('.json'):
        return load_json_problem(input_path)
    elif input_path.endswith('.qplib'):
        from tools.qplib_parser import parse_qplib_file
        return parse_qplib_file(input_path)
    elif input_path.endswith('.smt2'):
        from tools.smt2_parser import parse_smt2_file
        return parse_smt2_file(input_path)
    elif input_path.endswith('.cnf'):
        if k is not None and k >= 2:
            from tools.diversesat_parser import parse_diversesat
            return parse_diversesat(input_path, k=k)
        from tools.cnf_parser import parse_cnf_file
        return parse_cnf_file(input_path)
    else:
        try:
            return load_json_problem(input_path)
        except json.JSONDecodeError:
            from tools.qplib_parser import parse_qplib_file
            return parse_qplib_file(input_path)


def format_result(result, problem_name, time_read, time_total):
    encoding_str = result['encoding']
    if result.get('use_decomposition'):
        encoding_str += "+DECOMP"

    status = result.get('solver_status', 'UNKNOWN')

    timings = result.get('timings', {})
    time_encode = timings.get('build_total', 0.0)
    time_solve = timings.get('solve', 0.0)

    opt_display = result['objective_value'] if status in ('OPTIMAL', 'FEASIBLE') else 'N/A'

    return (
        f">>> Benchmark {problem_name} "
        f"Encoding {encoding_str} "
        f"Solver {result['solver']} "
        f"Status {status} "
        f"OPT {opt_display} "
        f"Cost {result['maxsat_cost']} "
        f"Vars {result['num_variables']} "
        f"Hard {result['num_hard_clauses']} "
        f"Soft {result['num_soft_clauses']} "
        f"TopW {result['top_weight']} "
        f"TimeRead {time_read:.3f} "
        f"TimeEncode {time_encode:.3f} "
        f"TimeSolve {time_solve:.3f} "
        f"TimeTotal {time_total:.3f}"
    )


def print_verbose(result, problem):
    print("\n" + "=" * 60)
    print("Solving Result Details")
    print("=" * 60)

    print(f"\n Encoding Statistics:")
    encoding_str = result['encoding']
    if result.get('use_decomposition'):
        encoding_str += " + Order Decomposition"
    print(f"  Encoding: {encoding_str}")
    print(f"  Variables: {result['num_variables']}")
    print(f"  Hard clauses: {result['num_hard_clauses']}")
    print(f"  Soft clauses: {result['num_soft_clauses']}")
    print(f"  Top weight: {result['top_weight']}")

    print(f"\n Solving Result:")
    print(f"  Solver: {result['solver']}")
    print(f"  Objective value: {result['objective_value']}")
    print(f"  MaxSAT cost: {result['maxsat_cost']}")
    print(f"  Total soft weight: {result['total_soft_weight']}")

    if 'timings' in result:
        print(f"\n Timing Statistics:")
        timings = result['timings']
        if 'preprocess' in timings:
            print(f"  Preprocessing: {timings['preprocess']*1000:.2f} ms")
        if 'domain_constraints' in timings:
            print(f"  Domain constraints: {timings['domain_constraints']*1000:.2f} ms")
        if 'problem_constraints' in timings:
            print(f"  Problem constraints: {timings['problem_constraints']*1000:.2f} ms")
        if 'objective_encoding' in timings:
            print(f"  Objective encoding: {timings['objective_encoding']*1000:.2f} ms")
        if 'build_total' in timings:
            print(f"  Build total: {timings['build_total']*1000:.2f} ms")
        if 'solve' in timings:
            print(f"  Solve time: {timings['solve']*1000:.2f} ms")
        if 'total' in timings:
            print(f"  Total time: {timings['total']*1000:.2f} ms")

    if 'assignment' in result and result['assignment']:
        print(f"\n Assignment:")
        print(f"  (assignment vector length: {len(result['assignment'])})")


def main():
    parser = argparse.ArgumentParser(
        description="NLIP to MaxSAT: convert nonlinear integer programming to MaxSAT and solve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py problem.json                    # solve with default settings
  python main.py problem.json -e UNA             # use UNA encoding
  python main.py problem.json -e BIN -s MAXHS    # use BIN encoding + MaxHS solver
  python main.py problem.json -e BIN --decomp    # use BIN encoding + Order Decomposition
  python main.py problem.json -o out.wcnf        # save WCNF file
  python main.py problem.json -v                 # verbose output
  python main.py formula.cnf --k 2 -e BIN        # Diverse-SAT: 2 models, BIN encoding

Encoding:
  OH  - One-Hot encoding (default, general-purpose)
  UNA - Unary encoding (good for 2D, smaller soft weights)
  BIN - Binary encoding (fewest variables)

Optimization:
  --decomp              use Order Decomposition for high-degree terms (BIN only)
  --decomp-strategy     sequential (default) or binary_tree decomposition
  --decomp-relaxed      use relaxed Mul sense (default: exact per paper)
  --no-decomp-shared    disable shared substructure cache (paper Figure 3c)
  --no-weight-gcd       disable soft weight GCD normalization

Solvers:
  RC2      - PySAT built-in solver (default)
  MAXHS    - MaxHS solver (ILP-based)
  WMAXCDCL - WMaxCDCL solver (Branch-and-Bound)
  OPENWBO  - OpenWBO solver (SAT-based)
        """
    )

    parser.add_argument(
        "input",
        help="input problem file (JSON, QPLIB, SMT2, or CNF format)"
    )

    parser.add_argument(
        "-e", "--encoding",
        choices=["OH", "UNA", "BIN"],
        default="OH",
        help="encoding method (default: OH)"
    )

    parser.add_argument(
        "--decomp",
        action="store_true",
        help="enable Order Decomposition for high-degree terms (BIN only)"
    )

    parser.add_argument(
        "--decomp-threshold",
        type=int,
        default=3,
        help="minimum degree to trigger decomposition (default: 3)"
    )

    parser.add_argument(
        "--decomp-strategy",
        choices=["sequential", "binary_tree"],
        default="sequential",
        help="decomposition strategy: sequential (paper §4 SD) or binary_tree (paper Figure 3b)"
    )

    parser.add_argument(
        "--decomp-relaxed",
        action="store_true",
        help="use relaxed sense in Mul (faster but less precise; default: exact per paper)"
    )

    parser.add_argument(
        "--no-decomp-shared",
        action="store_true",
        help="disable shared substructure cache (paper Figure 3c)"
    )

    parser.add_argument(
        "--no-weight-gcd",
        action="store_true",
        help="disable soft weight GCD normalization"
    )

    parser.add_argument(
        "-s", "--solver",
        choices=["RC2", "MAXHS", "WMAXCDCL", "OPENWBO", "NONE"],
        default="RC2",
        help="solver backend (default: RC2). Use NONE to only generate WCNF."
    )

    parser.add_argument(
        "-o", "--output",
        help="save WCNF file to specified path"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="verbose output mode"
    )

    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="disable preprocessing"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the correctness of solving result"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="timeout in seconds; prints TIMEOUT status and exits"
    )

    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Diverse-SAT: number of diverse models (k >= 2). "
             "Only applies to .cnf inputs; triggers quadratic objective modeling."
    )

    args = parser.parse_args()

    problem_name = os.path.basename(args.input)
    t_start = time.time()

    if args.timeout and hasattr(signal, 'SIGALRM'):
        _timeout_ctx.update({
            "problem_name": problem_name,
            "encoding": args.encoding,
            "solver": args.solver,
            "t_start": t_start,
        })

        def _on_timeout(signum, frame):
            elapsed = time.time() - _timeout_ctx["t_start"]
            print(f"\n>>> Benchmark {_timeout_ctx['problem_name']} "
                  f"Encoding {_timeout_ctx['encoding']} "
                  f"Solver {_timeout_ctx['solver']} "
                  f"Status TIMEOUT OPT N/A "
                  f"TimeTotal {elapsed:.3f}")
            sys.stdout.flush()
            os._exit(0)

        signal.signal(signal.SIGALRM, _on_timeout)
        signal.alarm(args.timeout)

    if not os.path.exists(args.input):
        print(f"Error: file not found - {args.input}", file=sys.stderr)
        print(f"\n>>> Benchmark {problem_name} "
              f"Encoding {args.encoding} Solver {args.solver} "
              f"Status ERROR OPT N/A TimeTotal 0.000")
        return 1

    if args.verbose:
        print(f"Loading problem file: {args.input}")

    try:
        problem = load_problem(args.input, k=args.k)
    except Exception as e:
        print(f"Error: failed to parse problem file - {e}", file=sys.stderr)
        time_total = time.time() - t_start
        print(f"\n>>> Benchmark {problem_name} "
              f"Encoding {args.encoding} Solver {args.solver} "
              f"Status ERROR OPT N/A TimeTotal {time_total:.3f}")
        return 1
    time_read = time.time() - t_start

    if args.verbose:
        print(f"  Variables: {len(problem.get('variables', {}))}")
        print(f"  Constraints: {len(problem.get('constraints', []))}")
        print(f"  Objective terms: {len(problem.get('objective', {}).get('terms', []))}")

    cfg = EncodingConfig()
    cfg.enable_preprocess = not args.no_preprocess
    cfg.use_decomposition = args.decomp
    cfg.decomp_threshold = args.decomp_threshold
    cfg.decomp_strategy = args.decomp_strategy
    cfg.decomp_exact = not args.decomp_relaxed
    cfg.decomp_shared = not args.no_decomp_shared
    cfg.weight_gcd_normalize = not args.no_weight_gcd
    if args.timeout:
        total_timeout = int(args.timeout)
        now = time.time()
        elapsed = now - t_start
        remaining = max(0.0, total_timeout - elapsed)
        # Allocate a conservative budget to encoding (PBEnc may be non-interruptible),
        # but keep most time for the MaxSAT solver to reduce TIMEOUTs.
        encoding_budget = min(1800.0, max(30.0, remaining * 0.25))
        # Ensure we keep some time for solving.
        if remaining - encoding_budget < 30.0:
            encoding_budget = max(0.0, remaining - 30.0)
        cfg.encoding_deadline = (now + encoding_budget) if encoding_budget > 0 else 0.0
        # Initial estimate; will be refreshed right before solving.
        cfg.external_solver_timeout = max(0, int(remaining - encoding_budget - 5.0))
    else:
        cfg.external_solver_timeout = 0

    if args.decomp and args.encoding != "BIN":
        print(f"Warning: --decomp only works with BIN encoding, "
              f"current is {args.encoding}, ignoring", file=sys.stderr)
        cfg.use_decomposition = False

    if args.output and args.solver == "NONE":
        if args.verbose:
            print(f"\nBuilding WCNF ({args.encoding} encoding)...")
        wcnf, name2idx, _vpool = build_wcnf(problem, args.encoding, cfg)
        wcnf.to_file(args.output)
        print(f"WCNF saved to: {args.output}")
        return 0

    if args.verbose:
        enc_desc = args.encoding
        if cfg.use_decomposition:
            enc_desc += (f" + Decomposition(threshold={cfg.decomp_threshold}, "
                         f"strategy={cfg.decomp_strategy}, "
                         f"exact={cfg.decomp_exact}, shared={cfg.decomp_shared})")
        opts = []
        if cfg.weight_gcd_normalize:
            opts.append("weight-GCD")
        if opts:
            enc_desc += f" [{', '.join(opts)}]"
        print(f"\nEncoding and solving (encoding: {enc_desc}, solver: {args.solver})...")

    try:
        # Refresh external solver timeout using the actual remaining wall-clock budget.
        if args.timeout:
            remaining = max(0.0, float(args.timeout) - (time.time() - t_start))
            cfg.external_solver_timeout = max(0, int(remaining - 5.0))
        result = build_and_solve(problem, args.encoding, cfg, solver=args.solver)
    except (EncodingTimeoutError, _EncodingTimeout) as e:
        time_total = time.time() - t_start
        print(f"  [encoding-timeout] {e}", file=sys.stderr)
        enc_str = args.encoding
        if cfg.use_decomposition:
            enc_str += "+DECOMP"
        print(f"\n>>> Benchmark {problem_name} "
              f"Encoding {enc_str} Solver {args.solver} "
              f"Status TIMEOUT OPT N/A TimeTotal {time_total:.3f}")
        return 1
    except Exception as e:
        print(f"Error: solving failed - {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        time_total = time.time() - t_start
        print(f"\n>>> Benchmark {problem_name} "
              f"Encoding {args.encoding} Solver {args.solver} "
              f"Status ERROR OPT N/A TimeTotal {time_total:.3f}")
        return 1
    time_total = time.time() - t_start

    if args.output:
        wcnf, _ = build_wcnf(problem, args.encoding, cfg)
        wcnf.to_file(args.output)
        if args.verbose:
            print(f"\nWCNF saved to: {args.output}")

    if args.verbose:
        print_verbose(result, problem)

    if args.verify:
        if args.verbose:
            print(f"\nVerifying solving result...")
        is_valid, report = verify_solution(problem, result, args.encoding)
        print("\n" + format_verification_report(report))

    print("\n" + format_result(result, problem_name, time_read, time_total))

    return 0


if __name__ == "__main__":
    sys.exit(main())