/**
 * @file Types.hpp
 * @brief Common type definitions for SEMSWS
 *
 * Consolidated enum types and their conversion functions.
 * Previously scattered across Config.hpp, OperatorFactory.hpp, Receiver.hpp, Material.hpp.
 */

#ifndef SEM_COMMON_TYPES_HPP
#define SEM_COMMON_TYPES_HPP

#include <string>
#include <mfem.hpp>

namespace SEM {

// =============================================================================
// Simulation Mode (moved from Config.hpp)
// =============================================================================
enum class SimulationMode {
    Forward,
    Inversion
};

// =============================================================================
// Domain Type (moved from Receiver.hpp)
// =============================================================================
enum class DomainType {
    Solid = 0,
    Fluid = 1
};

// =============================================================================
// Material Type (moved from Material.hpp)
// Integer values are fixed for backward compatibility (HDF5 files, etc.)
// =============================================================================
enum class MaterialType {
    IsotropicElastic = 1,
    IsotropicAcoustic = 2,
    AnisotropicElastic = 4,
    // Visco_IsotropicElastic = 5,
    // Visco_IsotropicAcoustic = 6
    Coupled = 22          // meta-type: fluid-solid coupled, see CoupledMaterialConfig
};

/// Get DomainType from MaterialType
/// Solid: IsotropicElastic, AnisotropicElastic (elastic materials)
/// Fluid: IsotropicAcoustic
/// Coupled is a meta-type that holds a fluid and a solid sub-material; callers
/// must query the nested sub-material's type instead of asking for a single
/// DomainType. Calling this with Coupled is a programming error.
inline DomainType GetDomainFromMaterial(MaterialType mat_type) {
    switch (mat_type) {
        case MaterialType::IsotropicElastic:
        case MaterialType::AnisotropicElastic:
            return DomainType::Solid;
        case MaterialType::IsotropicAcoustic:
            return DomainType::Fluid;
        case MaterialType::Coupled:
            MFEM_ABORT("GetDomainFromMaterial(Coupled) is invalid: Coupled holds "
                       "per-submesh materials. Dispatch on the nested fluid/solid "
                       "sub-material type instead.");
    }
    MFEM_ABORT("Unknown MaterialType enum value");
}

inline std::string MaterialTypeToString(MaterialType type) {
    switch (type) {
        case MaterialType::IsotropicElastic: return "isotropic_elastic";
        case MaterialType::IsotropicAcoustic: return "isotropic_acoustic";
        case MaterialType::Coupled: return "coupled";
        case MaterialType::AnisotropicElastic: return "anisotropic_elastic";
        // case MaterialType::Visco_IsotropicElastic: return "visco_isotropic_elastic";
        // case MaterialType::Visco_IsotropicAcoustic: return "visco_isotropic_acoustic";
        default: MFEM_ABORT("Unknown MaterialType enum value");
    }
}

inline MaterialType StringToMaterialType(const std::string& s) {
    if (s == "isotropic_elastic") return MaterialType::IsotropicElastic;
    if (s == "isotropic_acoustic") return MaterialType::IsotropicAcoustic;
    if (s == "coupled") return MaterialType::Coupled;
    if (s == "anisotropic_elastic") return MaterialType::AnisotropicElastic;
    // if (s == "visco_isotropic_elastic") return MaterialType::Visco_IsotropicElastic;
    // if (s == "visco_isotropic_acoustic") return MaterialType::Visco_IsotropicAcoustic;
    MFEM_ABORT("Unknown material type string: " + s);
}

// =============================================================================
// Receiver Type (moved from Receiver.hpp)
// Integer values are fixed for backward compatibility (HDF5 files, etc.)
// =============================================================================
enum class ReceiverType {
    Displacement = 1,  // DISP
    Velocity = 2,      // VEL
    Acceleration = 3,  // ACC
    Gradient = 4,      // GRAD (strain)
    Pressure = 10      // PS (acoustic) - value fixed for backward compatibility
};

inline std::string ReceiverTypeToString(ReceiverType type) {
    switch (type) {
        case ReceiverType::Displacement: return "DISP";
        case ReceiverType::Velocity: return "VEL";
        case ReceiverType::Acceleration: return "ACC";
        case ReceiverType::Gradient: return "GRAD";
        case ReceiverType::Pressure: return "PS";
        MFEM_ABORT("Unknown ReceiverType enum value");
    }
}

/// Dataset name used by the v2.0 HDF5 schema for this receiver type.
/// SEMSWS uses one canonical short form across all layers (YAML, HDF5,
/// ASCII suffix derivation), so this returns the same string as
/// `ReceiverTypeToString()` for supported types. Returns empty for types
/// that have no observed channel (e.g. Gradient) — writers must skip
/// those.
inline std::string ReceiverTypeToObservedChannel(ReceiverType type) {
    switch (type) {
        case ReceiverType::Displacement: return "DISP";
        case ReceiverType::Velocity:     return "VEL";
        case ReceiverType::Acceleration: return "ACC";
        case ReceiverType::Pressure:     return "PS";
        case ReceiverType::Gradient:     return "";
    }
    return "";
}

inline ReceiverType StringToReceiverType(const std::string& s) {
    // Short form only (DISP, VEL, ACC, PS, GRAD)
    if (s == "DISP") return ReceiverType::Displacement;
    if (s == "VEL") return ReceiverType::Velocity;
    if (s == "ACC") return ReceiverType::Acceleration;
    if (s == "PS") return ReceiverType::Pressure;
    if (s == "GRAD") return ReceiverType::Gradient;
    MFEM_ABORT("Unknown receiver type string: " + s + " \n(Valid: DISP, VEL, ACC, PS, GRAD)");
}

inline ReceiverType IntToReceiverType(int i) {
    switch (i) {
        case 1: return ReceiverType::Displacement;
        case 2: return ReceiverType::Velocity;
        case 3: return ReceiverType::Acceleration;
        case 4: return ReceiverType::Gradient;
        case 10: return ReceiverType::Pressure;
        MFEM_ABORT("Unknown ReceiverType integer value: " + std::to_string(i)); 
    }
}

// =============================================================================
// Wavelet Type
// =============================================================================
enum class WaveletType {
    Ricker,
    Gaussian,
    External
};

inline std::string WaveletTypeToString(WaveletType type) {
    switch (type) {
        case WaveletType::Ricker: return "ricker";
        case WaveletType::Gaussian: return "gaussian";
        case WaveletType::External: return "external";
        MFEM_ABORT("Unknown WaveletType enum value");
    }
}

inline WaveletType StringToWaveletType(const std::string& s) {
    if (s == "ricker") return WaveletType::Ricker;
    if (s == "gaussian") return WaveletType::Gaussian;
    if (s == "external") return WaveletType::External;
    MFEM_ABORT("Unknown wavelet type string: " + s);
}

// =============================================================================
// Source Type
// =============================================================================
enum class SourceType {
    Force,
    Pressure,
    MomentTensor
};

inline std::string SourceTypeToString(SourceType type) {
    switch (type) {
        case SourceType::Force: return "force";
        case SourceType::Pressure: return "pressure";
        case SourceType::MomentTensor: return "moment_tensor";
        MFEM_ABORT("Unknown SourceType enum value");
    }
}

inline SourceType StringToSourceType(const std::string& s) {
    if (s == "force") return SourceType::Force;
    if (s == "pressure") return SourceType::Pressure;
    if (s == "moment_tensor") return SourceType::MomentTensor;
    MFEM_ABORT("Unknown source type string: " + s);
}

}  // namespace SEM

#endif  // SEM_COMMON_TYPES_HPP
