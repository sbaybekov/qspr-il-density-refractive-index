"""Backward-compatible entry point: ``python qspr.py``.

The real implementation lives in :mod:`qspr_il.cli`; this file exists so the
command documented in the README keeps working.
"""

from qspr_il.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
