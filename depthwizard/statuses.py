"""Explicit machine-readable status/error codes for the geospatial pipeline.

The absolute-DSM workflow must NEVER silently convert a failed operation
into a fake result. Every failure mode that has a name gets a name here,
raised as :class:`DWStatusError` (a ValueError subclass so existing
``pytest.raises(ValueError)`` call sites and callers keep working).

Statuses are also echoed into payload/report metadata as plain strings so
the API/frontend can branch on them without parsing human prose.
"""

from __future__ import annotations

from typing import Optional


class DWStatusError(ValueError):
    """A pipeline failure with a machine-readable status code.

    ``code`` is one of the module-level constants below; ``detail`` carries
    the human explanation. ``str(err)`` includes both.
    """

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"[{code}] {detail}")


# --- georeference requirements -------------------------------------------
CRS_REQUIRED = "CRS_REQUIRED"
INVALID_GEOREFERENCE = "INVALID_GEOREFERENCE"
VERTICAL_REFERENCE_UNKNOWN = "VERTICAL_REFERENCE_UNKNOWN"

# --- DEM acquisition / alignment ------------------------------------------
DEM_REQUIRED = "DEM_REQUIRED"
DEM_UNAVAILABLE = "DEM_UNAVAILABLE"
DEM_COVERAGE_INSUFFICIENT = "DEM_COVERAGE_INSUFFICIENT"

# --- evaluation ------------------------------------------------------------
REFERENCE_DSM_REQUIRED_FOR_EVALUATION = "REFERENCE_DSM_REQUIRED_FOR_EVALUATION"
GRID_MISMATCH = "GRID_MISMATCH"

# --- geometry ---------------------------------------------------------------
INVALID_POLYGON = "INVALID_POLYGON"

ALL_CODES = (
    CRS_REQUIRED,
    INVALID_GEOREFERENCE,
    VERTICAL_REFERENCE_UNKNOWN,
    DEM_REQUIRED,
    DEM_UNAVAILABLE,
    DEM_COVERAGE_INSUFFICIENT,
    REFERENCE_DSM_REQUIRED_FOR_EVALUATION,
    GRID_MISMATCH,
    INVALID_POLYGON,
)


def status_of(err: BaseException) -> Optional[str]:
    """The status code of a DWStatusError, else None (never guesses)."""
    return getattr(err, "code", None)
