// SEMSWS HDF5 source/receiver schema (format_version 2.0).
// Mirrored on the driver side at driver/src/semsws_driver/io/hdf5_schema.py.

#ifndef SEM_HDF5_IO_SCHEMA_HPP
#define SEM_HDF5_IO_SCHEMA_HPP

#include <cstdio>
#include <string>

namespace SEM {
namespace HDF5Schema {

// -----------------------------------------------------------------------------
// Format version
// -----------------------------------------------------------------------------

/// Canonical version string written by the writer.
inline constexpr const char* kFormatVersion = "2.0";

/// Major part the reader requires. "1.x" files are rejected; users must
/// run `driver/tools/migrate_observed_v1_to_v2.py` to upgrade.
inline constexpr const char* kFormatVersionMajor = "2";

// -----------------------------------------------------------------------------
// Root attributes
// -----------------------------------------------------------------------------

inline constexpr const char* kAttrFormatVersion = "format_version";
inline constexpr const char* kAttrDt            = "dt";
inline constexpr const char* kAttrT0            = "t0";
inline constexpr const char* kAttrNSamples      = "n_samples";
inline constexpr const char* kAttrSpaceDim      = "space_dim";
inline constexpr const char* kAttrCoordSystem   = "coord_system";
inline constexpr const char* kAttrUnits         = "units";
inline constexpr const char* kAttrCreatedBy     = "created_by";
inline constexpr const char* kAttrCreatedAt     = "created_at";

inline constexpr const char* kDefaultCoordSystem = "cartesian";
inline constexpr const char* kDefaultUnits       = "SI";

// -----------------------------------------------------------------------------
// Top-level group names
// -----------------------------------------------------------------------------

inline constexpr const char* kGroupShots     = "shots";
inline constexpr const char* kGroupSources   = "sources";
inline constexpr const char* kGroupReceivers = "receivers";

// -----------------------------------------------------------------------------
// Per-element attribute / dataset names (sources & receivers)
// -----------------------------------------------------------------------------

inline constexpr const char* kAttrShotId    = "shot_id";
inline constexpr const char* kAttrId        = "id";
inline constexpr const char* kAttrLabel     = "label";
inline constexpr const char* kAttrType      = "type";
inline constexpr const char* kAttrPosition  = "position";
inline constexpr const char* kAttrDirection = "direction";

inline constexpr const char* kDatasetStf          = "stf";
inline constexpr const char* kDatasetMomentTensor = "moment_tensor";
inline constexpr const char* kAttrComponentOrder  = "component_order";
inline constexpr const char* kAttrMomentUnit      = "moment_unit";
inline constexpr const char* kDefaultMomentUnit   = "N*m";

// Source `@type` enum strings.
inline constexpr const char* kSourceTypeForce         = "force";
inline constexpr const char* kSourceTypePressure      = "pressure";
inline constexpr const char* kSourceTypeMomentTensor  = "moment_tensor";

// Receiver channel names. SEMSWS uses one canonical short form across all
// layers — YAML (`receivers.type: [VEL, DISP, ...]`), HDF5 channel datasets
// (`/shots/.../receivers/<r>/VEL`), `@types` attribute lists, and the
// `ReceiverType` enum string round-trip via `ReceiverTypeToString` /
// `StringToReceiverType` in include/common/Types.hpp. No fullword
// alternative exists in the v2.0 schema.
inline constexpr const char* kChannelPressure     = "PS";
inline constexpr const char* kChannelVelocity     = "VEL";
inline constexpr const char* kChannelDisplacement = "DISP";
inline constexpr const char* kChannelAcceleration = "ACC";

// Weight datasets are named `weight_<channel>`.
inline constexpr const char* kWeightPrefix = "weight_";

// -----------------------------------------------------------------------------
// Group-name builders (4-digit zero-padding; sortable, fixed-width)
// -----------------------------------------------------------------------------

inline std::string ShotKey(int shot_id) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "%04d", shot_id);
    return buf;
}

/// Full POSIX-style path to a shot group: "shots/NNNN".
inline std::string ShotGroupPath(int shot_id) {
    return std::string(kGroupShots) + "/" + ShotKey(shot_id);
}

/// Source group key (used inside `/shots/.../sources/`). 1-indexed.
inline std::string SourceKey(int id) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "S%04d", id);
    return buf;
}

/// Receiver group key (used inside `/shots/.../receivers/`). 1-indexed.
inline std::string ReceiverKey(int id) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "R%04d", id);
    return buf;
}

}  // namespace HDF5Schema
}  // namespace SEM

#endif  // SEM_HDF5_IO_SCHEMA_HPP
