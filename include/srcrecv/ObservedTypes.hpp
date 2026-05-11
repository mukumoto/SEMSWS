/**
 * @file ObservedTypes.hpp
 * @brief Shared helpers for observed-data type strings and channel names.
 *
 * Central place for:
 *   - ParseObservedType: YAML/HDF5 lowercase string -> ReceiverType
 *   - Channel-name <-> ReceiverType mapping for HDF5 `/receivers/Rxxx/<channel>`
 *   - Vector/scalar kind query
 */

#ifndef SEM_OBSERVED_TYPES_HPP
#define SEM_OBSERVED_TYPES_HPP

#include <mfem.hpp>
#include <string>
#include "common/Types.hpp"

namespace SEM {

// Channel names for the v2.0 HDF5 schema. SEMSWS uses one canonical short
// form (YAML, HDF5, ASCII suffix derivation all share these strings).
// Keep in sync with HDF5IOSchema.hpp.
inline const char* ObservedChannelName(ReceiverType t) {
    switch (t) {
        case ReceiverType::Pressure:     return "PS";
        case ReceiverType::Velocity:     return "VEL";
        case ReceiverType::Displacement: return "DISP";
        case ReceiverType::Acceleration: return "ACC";
        default: break;
    }
    MFEM_ABORT("ObservedChannelName: unsupported ReceiverType "
               << static_cast<int>(t));
    return "";
}

inline ReceiverType ParseObservedType(const std::string& s) {
    if (s == "PS")   return ReceiverType::Pressure;
    if (s == "VEL")  return ReceiverType::Velocity;
    if (s == "DISP") return ReceiverType::Displacement;
    if (s == "ACC")  return ReceiverType::Acceleration;
    MFEM_ABORT("Unknown observed type: " << s
               << " (valid: PS, VEL, DISP, ACC)");
    return ReceiverType::Pressure;
}

inline bool IsVectorObservedType(ReceiverType t) {
    return t == ReceiverType::Velocity
        || t == ReceiverType::Displacement
        || t == ReceiverType::Acceleration;
}

inline bool IsScalarObservedType(ReceiverType t) {
    return t == ReceiverType::Pressure;
}

}  // namespace SEM

#endif  // SEM_OBSERVED_TYPES_HPP
