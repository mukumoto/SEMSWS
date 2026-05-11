"""Cross-check that Python schema constants match C++ HDF5IOSchema.hpp.

The C++ header is the source-of-truth in builds; this test parses its
`inline constexpr const char* kFoo = "bar";` lines and asserts every
constant is present (and identical) in the Python `hdf5_schema` module.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from semsws_driver.io import hdf5_schema as S

REPO_ROOT = Path(__file__).resolve().parents[3]
HEADER = REPO_ROOT / "include" / "srcrecv" / "HDF5IOSchema.hpp"

# Each entry: (C++ symbol fragment, Python attribute name).
CONSTANT_PAIRS = [
    ("kFormatVersion",          "FORMAT_VERSION"),
    ("kFormatVersionMajor",     "FORMAT_VERSION_MAJOR"),
    ("kAttrFormatVersion",      "ATTR_FORMAT_VERSION"),
    ("kAttrDt",                 "ATTR_DT"),
    ("kAttrT0",                 "ATTR_T0"),
    ("kAttrNSamples",           "ATTR_N_SAMPLES"),
    ("kAttrSpaceDim",           "ATTR_SPACE_DIM"),
    ("kAttrCoordSystem",        "ATTR_COORD_SYSTEM"),
    ("kAttrUnits",              "ATTR_UNITS"),
    ("kAttrCreatedBy",          "ATTR_CREATED_BY"),
    ("kAttrCreatedAt",          "ATTR_CREATED_AT"),
    ("kDefaultCoordSystem",     "DEFAULT_COORD_SYSTEM"),
    ("kDefaultUnits",           "DEFAULT_UNITS"),
    ("kGroupShots",             "GROUP_SHOTS"),
    ("kGroupSources",           "GROUP_SOURCES"),
    ("kGroupReceivers",         "GROUP_RECEIVERS"),
    ("kAttrShotId",             "ATTR_SHOT_ID"),
    ("kAttrId",                 "ATTR_ID"),
    ("kAttrLabel",              "ATTR_LABEL"),
    ("kAttrType",               "ATTR_TYPE"),
    ("kAttrPosition",           "ATTR_POSITION"),
    ("kAttrDirection",          "ATTR_DIRECTION"),
    ("kDatasetStf",             "DATASET_STF"),
    ("kDatasetMomentTensor",    "DATASET_MOMENT_TENSOR"),
    ("kAttrComponentOrder",     "ATTR_COMPONENT_ORDER"),
    ("kAttrMomentUnit",         "ATTR_MOMENT_UNIT"),
    ("kDefaultMomentUnit",      "DEFAULT_MOMENT_UNIT"),
    ("kSourceTypeForce",        "SOURCE_TYPE_FORCE"),
    ("kSourceTypePressure",     "SOURCE_TYPE_PRESSURE"),
    ("kSourceTypeMomentTensor", "SOURCE_TYPE_MOMENT_TENSOR"),
    ("kChannelPressure",        "CHANNEL_PRESSURE"),
    ("kChannelVelocity",        "CHANNEL_VELOCITY"),
    ("kChannelDisplacement",    "CHANNEL_DISPLACEMENT"),
    ("kChannelAcceleration",    "CHANNEL_ACCELERATION"),
    ("kWeightPrefix",           "WEIGHT_PREFIX"),
]

# Pattern matches `inline constexpr const char* kFoo = "bar";`. The
# `inline` may be on a previous line wrapped, but in our header it's
# all on one line.
_RE = re.compile(
    r"inline\s+constexpr\s+const\s+char\*\s+(\w+)\s*=\s*\"([^\"]*)\"\s*;"
)


@pytest.fixture(scope="module")
def cpp_constants() -> dict[str, str]:
    if not HEADER.exists():
        pytest.skip(f"C++ header not found: {HEADER}")
    text = HEADER.read_text(encoding="utf-8")
    return dict(_RE.findall(text))


def test_header_was_parsed(cpp_constants):
    # If parsing failed entirely we'd silently match zero pairs, so
    # explicitly assert we found enough constants.
    assert len(cpp_constants) >= len(CONSTANT_PAIRS), (
        f"parsed only {len(cpp_constants)} constants from {HEADER} "
        f"(expected ≥ {len(CONSTANT_PAIRS)})"
    )


@pytest.mark.parametrize("cpp_name,py_name", CONSTANT_PAIRS,
                         ids=[p[1] for p in CONSTANT_PAIRS])
def test_constant_matches(cpp_constants, cpp_name: str, py_name: str):
    assert cpp_name in cpp_constants, (
        f"missing C++ constant {cpp_name} in {HEADER}"
    )
    py_value = getattr(S, py_name)
    cpp_value = cpp_constants[cpp_name]
    assert cpp_value == py_value, (
        f"{cpp_name} (C++: {cpp_value!r}) != {py_name} "
        f"(Python: {py_value!r})"
    )
