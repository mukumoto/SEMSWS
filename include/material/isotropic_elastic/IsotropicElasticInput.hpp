/**
 * @file IsotropicElasticInput.hpp
 * @brief Input data structures for isotropic elastic material creation
 *
 * This file defines the intermediate data structures used between
 * loading (Loader) and construction (Builder) of IsotropicElasticMaterial.
 *
 * The separation allows:
 * - Clean validation of input data
 * - Testing Loader and Builder independently
 * - Future support for different data sources
 */

#ifndef SEM_ISOTROPIC_ELASTIC_INPUT_HPP
#define SEM_ISOTROPIC_ELASTIC_INPUT_HPP

#include "material/MaterialUtils.hpp"
#include "material/MaterialField.hpp"
#include "material/isotropic_elastic/IsotropicElasticIO.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace SEM {

// =============================================================================
// Constant Parameters (uniform material)
// =============================================================================

/**
 * @struct IsotropicElasticConstantParams
 * @brief Uniform isotropic elastic material parameters
 *
 * Used when config.format == "constant".
 * All elements have the same material properties.
 */
struct IsotropicElasticConstantParams {
    real_t vp;          ///< P-wave velocity (m/s)
    real_t vs;          ///< S-wave velocity (m/s)
    real_t rho;         ///< Density (kg/m^3)
    real_t qkappa = 0;  ///< Q factor for bulk modulus (0 = no attenuation)
    real_t qmu = 0;     ///< Q factor for shear modulus (0 = no attenuation)
};

// =============================================================================
// Grid Parameters (spatially-varying material from file)
// =============================================================================

// Note: ElasticGridParams uses the IO structures from IsotropicElasticIO.hpp
// - IsotropicElasticMaterialData2D for 2D
// - IsotropicElasticMaterialData3D for 3D
// This avoids duplicating the grid data structure.

// =============================================================================
// ByAttribute Parameters (material per mesh attribute)
// =============================================================================

// Note: Uses IsotropicElasticAttributeEntry from IsotropicElasticIO.hpp

// =============================================================================
// ByAttributeMixed Parameters (mixed constant/grid per attribute)
// =============================================================================

/**
 * @struct IsotropicElasticMixedAttributeEntry
 * @brief Entry for mixed attribute material (constant or grid per attribute)
 *
 * Used when config.format == "by_attribute_mixed".
 * Each attribute can have either constant values or grid-interpolated values.
 */
struct IsotropicElasticMixedAttributeEntry {
    int attribute;              ///< Mesh element attribute number
    /// Sub-mode: "constant" | "grid" | "adios2"
    std::string mode = "constant";
    /// Backward-compat: true iff mode == "grid" (set by Loader)
    bool is_heterogeneous = false;

    // Constant mode parameters
    real_t vp = 0.0;            ///< P-wave velocity (m/s)
    real_t vs = 0.0;            ///< S-wave velocity (m/s)
    real_t rho = 0.0;           ///< Density (kg/m^3)
    real_t Qkappa = -1.0;       ///< Q factor for bulk modulus (-1 = no attenuation)
    real_t Qmu = -1.0;          ///< Q factor for shear modulus (-1 = no attenuation)

    // Grid mode parameters (loaded from ascii file)
    std::optional<IsotropicElasticMaterialData2D> grid_data_2d;
    std::optional<IsotropicElasticMaterialData3D> grid_data_3d;

    // ADIOS2 mode parameters (pre-computed GLL data from .bp files)
    std::shared_ptr<MaterialField> adios2_vp;
    std::shared_ptr<MaterialField> adios2_vs;
    std::shared_ptr<MaterialField> adios2_rho;
    std::shared_ptr<MaterialField> adios2_qkappa;   // visco-elastic
    std::shared_ptr<MaterialField> adios2_qmu;      // visco-elastic
};

// =============================================================================
// Combined Input Structure
// =============================================================================

/**
 * @struct IsotropicElasticInput
 * @brief Unified input structure for creating IsotropicElasticMaterial
 *
 * This structure holds all the information needed to create an
 * IsotropicElasticMaterial, regardless of the data source.
 * Exactly one of the optional fields should be set based on `source`.
 *
 * Usage:
 * 1. Loader reads config and populates this structure
 * 2. Builder uses this structure to create the Material
 *
 * This design keeps Material creation logic clean and testable.
 */
struct IsotropicElasticInput {
    DataSource source;  ///< Which data source is used

    // Exactly one of these should be set based on `source`
    std::optional<IsotropicElasticConstantParams> constant;                      ///< For DataSource::Constant
    std::optional<IsotropicElasticMaterialData2D> grid_2d;              ///< For DataSource::Grid (2D)
    std::optional<IsotropicElasticMaterialData3D> grid_3d;              ///< For DataSource::Grid (3D)
    std::optional<std::vector<IsotropicElasticAttributeEntry>> by_attribute;  ///< For DataSource::ByAttribute
    std::optional<std::vector<IsotropicElasticMixedAttributeEntry>> by_attribute_mixed;  ///< For DataSource::ByAttributeMixed

    // ADIOS2: pre-computed GLL data (vp, vs, rho, optionally qkappa/qmu as MaterialField)
    std::shared_ptr<MaterialField> adios2_vp;      ///< For DataSource::ADIOS2
    std::shared_ptr<MaterialField> adios2_vs;      ///< For DataSource::ADIOS2
    std::shared_ptr<MaterialField> adios2_rho;     ///< For DataSource::ADIOS2
    std::shared_ptr<MaterialField> adios2_qkappa;  ///< For DataSource::ADIOS2 (visco-elastic)
    std::shared_ptr<MaterialField> adios2_qmu;     ///< For DataSource::ADIOS2 (visco-elastic)

    // Attenuation settings (shared across all sources)
    MaterialAttenuationConfig attenuation;

    // Dimension info (set by Loader for Grid/ByAttribute, computed for Constant)
    bool is_3d = false;

    /// Check if the input structure is valid
    bool IsValid() const {
        switch (source) {
            case DataSource::Constant:
                return constant.has_value();
            case DataSource::Grid:
                return is_3d ? grid_3d.has_value() : grid_2d.has_value();
            case DataSource::ByAttribute:
                return by_attribute.has_value() && !by_attribute->empty();
            case DataSource::ByAttributeMixed:
                return by_attribute_mixed.has_value() && !by_attribute_mixed->empty();
            case DataSource::ADIOS2:
                return adios2_vp && adios2_vs && adios2_rho;
        }
        return false;
    }

    /// Check if attenuation is enabled
    bool HasAttenuation() const {
        return attenuation.enabled;
    }
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ELASTIC_INPUT_HPP
