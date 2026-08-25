"""pytest bootstrap for the benchmark test tier.

Ensures the repo root is importable (so `import data._utils` / `import benchmark...`
work regardless of how pytest is invoked) and auto-skips torch-marked tests when
torch is unavailable (the `preprocess` env has no torch; the `vcell` env does).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest  # noqa: E402


def pytest_collection_modifyitems(config, items):
    try:
        import torch  # noqa: F401
        return
    except Exception:
        pass
    skip_torch = pytest.mark.skip(reason="needs torch (run in the vcell env)")
    for item in items:
        if "torch" in item.keywords:
            item.add_marker(skip_torch)
