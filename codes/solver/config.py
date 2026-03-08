import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SOLVER_DIR = str(_PROJECT_ROOT / "solvers" / "maxsat")


@dataclass
class EncodingConfig:
    encoding: str = "OH"

    use_decomposition: bool = False
    decomp_threshold: int = 3
    decomp_strategy: str = "sequential"  
    decomp_exact: bool = True            
    decomp_shared: bool = True            

    # Soft weight normalization
    weight_gcd_normalize: bool = True     # Divide all soft weights by their GCD (core-friendly)

    save_wcnf: bool = True
    workdir: str = os.environ.get("NLIP_WORKDIR", tempfile.gettempdir())

    maxhs_path: str = os.environ.get("NLIP_MAXHS", os.path.join(_SOLVER_DIR, "maxhs"))
    wmaxcdcl_path: str = os.environ.get("NLIP_WMAXCDCL", os.path.join(_SOLVER_DIR, "wmaxcdcl"))
    openwbo_path: str = os.environ.get("NLIP_OPENWBO", os.path.join(_SOLVER_DIR, "openwbo"))
    external_solver_timeout: int = 0

    # Preprocessing config
    enable_preprocess: bool = True
    preprocess_integerize: bool = True
    preprocess_scale_limit: int = 1000000

    # When enabled, encoding uses mapping y = x - lb (OH/UNA/BIN)
    use_mapping_shift: bool = False

    # Encoding-phase deadline (absolute time.time() value); 0 = no deadline
    encoding_deadline: float = 0.0
