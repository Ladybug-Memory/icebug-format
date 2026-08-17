"""Shared lazy-import helper for the optional ``datafusion`` dependency.

datafusion is gated behind the ``convert-datafusion`` extra so that the base
package can be imported on a bare install.  :mod:`cli` (CSR graph conversion
from vertex/edge Parquet pairs) imports datafusion through this single helper,
which raises an actionable error when the dependency is missing rather than a
raw ``ModuleNotFoundError``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    import datafusion as _datafusion_module


def require_datafusion(context: str = "this feature") -> "_datafusion_module":
    """Lazily import and return the ``datafusion`` module.

    Args:
        context: Short phrase describing what needs datafusion (e.g.
            ``"the 'convert-datafusion' feature"``), used in the error message.

    Raises:
        ImportError: With install instructions for the ``convert-datafusion``
            extra when datafusion is not installed.
    """
    try:
        import datafusion
    except ImportError as exc:  # pragma: no cover - exercised when extra absent
        raise ImportError(
            f"datafusion is required by {context} but is not installed. "
            "Install it with:\n"
            '    pip install "icebug-format[convert-datafusion]"'
        ) from exc
    return datafusion
