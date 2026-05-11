"""SEMSWS v2.0 HDF5 schema constants. Mirrors `include/srcrecv/HDF5IOSchema.hpp`
and is kept in sync by `tests/io/test_hdf5_schema_consistency.py`.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Format version
# ---------------------------------------------------------------------------

FORMAT_VERSION = "2.0"
FORMAT_VERSION_MAJOR = "2"

# ---------------------------------------------------------------------------
# Root attribute names
# ---------------------------------------------------------------------------

ATTR_FORMAT_VERSION = "format_version"
ATTR_DT             = "dt"
ATTR_T0             = "t0"
ATTR_N_SAMPLES      = "n_samples"
ATTR_SPACE_DIM      = "space_dim"
ATTR_COORD_SYSTEM   = "coord_system"
ATTR_UNITS          = "units"
ATTR_CREATED_BY     = "created_by"
ATTR_CREATED_AT     = "created_at"

DEFAULT_COORD_SYSTEM = "cartesian"
DEFAULT_UNITS        = "SI"

# ---------------------------------------------------------------------------
# Top-level group names
# ---------------------------------------------------------------------------

GROUP_SHOTS     = "shots"
GROUP_SOURCES   = "sources"
GROUP_RECEIVERS = "receivers"

# ---------------------------------------------------------------------------
# Per-element attribute / dataset names
# ---------------------------------------------------------------------------

ATTR_SHOT_ID    = "shot_id"
ATTR_ID         = "id"
ATTR_LABEL      = "label"
ATTR_TYPE       = "type"
ATTR_POSITION   = "position"
ATTR_DIRECTION  = "direction"

DATASET_STF             = "stf"
DATASET_MOMENT_TENSOR   = "moment_tensor"
ATTR_COMPONENT_ORDER    = "component_order"
ATTR_MOMENT_UNIT        = "moment_unit"
DEFAULT_MOMENT_UNIT     = "N*m"

# Source `@type` enum strings.
SOURCE_TYPE_FORCE         = "force"
SOURCE_TYPE_PRESSURE      = "pressure"
SOURCE_TYPE_MOMENT_TENSOR = "moment_tensor"

# Receiver channel names (canonical short form, matches YAML / C++ enum).
CHANNEL_PRESSURE     = "PS"
CHANNEL_VELOCITY     = "VEL"
CHANNEL_DISPLACEMENT = "DISP"
CHANNEL_ACCELERATION = "ACC"

WEIGHT_PREFIX = "weight_"

# Canonical moment-tensor component orders.
MT_COMPONENT_ORDER_3D = ["Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"]
MT_COMPONENT_ORDER_2D = ["Mxx", "Myy", "Mxy"]


# ---------------------------------------------------------------------------
# Group-name builders (4-digit zero-padding; sortable, fixed-width)
# ---------------------------------------------------------------------------


def shot_key(shot_id: int) -> str:
    return f"{shot_id:04d}"


def shot_group_path(shot_id: int) -> str:
    return f"{GROUP_SHOTS}/{shot_key(shot_id)}"


def source_key(source_id: int) -> str:
    return f"S{source_id:04d}"


def receiver_key(receiver_id: int) -> str:
    return f"R{receiver_id:04d}"
