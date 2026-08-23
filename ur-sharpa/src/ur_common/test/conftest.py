"""Make `ur_common` and the sibling test modules importable without installing.

Lets `python3 -m pytest test -q` work from the package root, and `colcon test`
work from the workspace.
"""

import os
import sys

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_TEST_DIR)

_PATHS = [_PKG_ROOT, _TEST_DIR]

# Add the repo's `interface/` if this package still sits inside the checkout, so
# the joint-order contract test can compare against the frozen interface order
# instead of skipping. Harmless once ur_common is installed standalone.
_INTERFACE = os.path.normpath(
    os.path.join(_PKG_ROOT, "..", "..", "..", "interface")
)
if os.path.isdir(os.path.join(_INTERFACE, "sharpa_interface")):
    _PATHS.append(_INTERFACE)

for path in _PATHS:
    if path not in sys.path:
        sys.path.insert(0, path)
