# ==============================================================================
# Local Mock Configuration for SReT-ToMe Profiling Integration
# ==============================================================================

import time
from pathlib import Path
from types import SimpleNamespace

def get_params():
    """
    Stubs out the expected configuration flags and save anchors required by toolbox.py and perf_monitor.py.
    """
    return SimpleNamespace(
        flags=SimpleNamespace(
            log_to_terminal=False,
            log_to_file=False,
            debug_flag=False
        ),
        logistics=SimpleNamespace(
            anchor_path=Path(".") 
        ),
        constants=SimpleNamespace(
            master_seed=42
        )
    )

def get_runtime():
    """Provides a clean, timestamp-based ID for naming output files."""
    return SimpleNamespace(
        execution_id=time.strftime("%Y%m%d_%H%M%S")
    )