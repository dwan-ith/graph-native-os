"""
pytest configuration for tiny-os.

Ensures `compiler` imports resolve regardless of invocation directory and
gives Windows users an escape hatch for broken system temp dirs
(pytest --basetemp can be passed on the command line instead).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
