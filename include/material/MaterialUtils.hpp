/**
 * @file MaterialUtils.hpp
 * @brief Common utility functions and types for material creation
 *
 * Provides shared utilities used across different material types:
 * - DataSource: Enum for material data format
 * - MaterialAttenuationConfig: Common attenuation configuration
 * - VelocityToLame: Convert seismic velocities to Lame parameters
 */

#ifndef SEM_MATERIAL_UTILS_HPP
#define SEM_MATERIAL_UTILS_HPP

#include <mfem.hpp>
#include <utility>

namespace SEM {

using mfem::real_t;

// =============================================================================
// Data Source Type
// =============================================================================

/**
 * @enum DataSource
 * @brief Specifies how material data is provided
 */
enum class DataSource {
    Constant,         ///< Uniform material properties
    Grid,             ///< Spatially-varying data on regular grid (ascii format)
    ByAttribute,      ///< Properties assigned by mesh element attribute (constant per attribute)
    ByAttributeMixed, ///< Mixed: some attributes constant, some from grid files
    ADIOS2            ///< Pre-computed GLL data from ADIOS2 .bp files (FWI iterations)
};

// =============================================================================
// Attenuation Configuration (for material creation)
// =============================================================================

/**
 * @struct MaterialAttenuationConfig
 * @brief Common attenuation configuration parameters for material creation
 *
 * Used by Loader/Builder to configure viscoelastic attenuation.
 * Note: This is different from AttenuationParams in AttenuationCoeffs.hpp
 * which contains computed coefficients (tau_epsilon, tau_sigma, etc.).
 */
struct MaterialAttenuationConfig {
    bool enabled = false;       ///< Whether attenuation is enabled
    real_t f0 = 1.0;            ///< Reference frequency for Q fitting (Hz)
    int n_units = 3;            ///< Number of SLS relaxation mechanisms
    real_t qkappa = 0.0;        ///< Q factor for bulk modulus (if constant)
    real_t qmu = 0.0;           ///< Q factor for shear modulus (elastic only, if constant)
};

// =============================================================================
// Velocity Conversion Utilities
// =============================================================================

/**
 * @brief Convert P-wave and S-wave velocities to Lame parameters
 *
 * @param vp P-wave velocity (m/s)
 * @param vs S-wave velocity (m/s)
 * @param rho Density (kg/m^3)
 * @return {lambda, mu} Lame parameters (Pa)
 */
inline std::pair<real_t, real_t> VelocityToLame(real_t vp, real_t vs, real_t rho) {
    real_t mu = rho * vs * vs;
    real_t lambda = rho * vp * vp - 2.0 * mu;
    return {lambda, mu};
}

/**
 * @brief Compute bulk modulus from P-wave velocity
 *
 * @param vp P-wave velocity (m/s)
 * @param rho Density (kg/m^3)
 * @return kappa Bulk modulus (Pa)
 *
 * Note: For acoustic media, kappa = rho * vp^2
 */
inline real_t VelocityToKappa(real_t vp, real_t rho) {
    return rho * vp * vp;
}

/**
 * @brief Compute bulk modulus from Lame parameters
 *
 * @param lambda Lame's first parameter (Pa)
 * @param mu Shear modulus (Pa)
 * @return kappa Bulk modulus (Pa)
 *
 * Note: For 2D plane strain, kappa = lambda + mu (not lambda + 2/3*mu)
 */
inline real_t LameToKappa(real_t lambda, real_t mu) {
    return lambda + mu;
}

}  // namespace SEM

#endif  // SEM_MATERIAL_UTILS_HPP
