"""Shared lazy-import helper for the optional ``duckdb`` dependency.

duckdb is gated behind the ``convert`` (and ``convert-duckdb``) extras so that
the base package can be imported on a bare install.  Both :mod:`cli` (CSR
graph conversion) and :mod:`memory` (:meth:`IcebugMemGraph.from_arrow_tables`)
import duckdb through this single helper, which raises an actionable error when
the dependency is missing rather than a raw ``ModuleNotFoundError``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    import duckdb as _duckdb_module


def require_duckdb(context: str = "this feature") -> "_duckdb_module":
    """Lazily import and return the ``duckdb`` module.

    Args:
        context: Short phrase describing what needs duckdb (e.g.
            ``"the 'convert' feature"`` or
            ``"IcebugMemGraph.from_arrow_tables()"``), used in the error message.

    Raises:
        ImportError: With install instructions for the ``convert`` /
            ``convert-duckdb`` extras when duckdb is not installed.
    """
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - exercised when extra absent
        raise ImportError(
            f"duckdb is required by {context} but is not installed. "
            "Install it with:\n"
            '    pip install "icebug-format[convert]"\n'
            "or, for just duckdb:\n"
            '    pip install "icebug-format[convert-duckdb]"'
        ) from exc
    return duckdb
