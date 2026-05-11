/**
 * @file IsotropicAcousticInput.hpp
 * @brief Input data structures for isotropic acoustic material creation
 *
 * This file defines the intermediate data structures used between
 * loading (Loader) and construction (Builder) of IsotropicAcousticMaterial.
 *
 * Note: Acoustic materials have no shear wave (vs) and no Qmu.
 */

#ifndef SEM_ISOTROPIC_ACOUSTIC_INPUT_HPP
#define SEM_ISOTROPIC_ACOUSTIC_INPUT_HPP

#include "material/MaterialUtils.hpp"
#include "material/MaterialField.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticIO.hpp"
#include <optional>
#include <vector>
#include <memory>

namespace SEM {

// =============================================================================
// Constant Parameters (uniform acoustic material)
// =============================================================================

/**
 * @struct IsotropicAcousticConstantParams
 * @brief Uniform isotropic acoustic material parameters
 *
 * Used when config.format == "constant".
 * Note: No vs (no shear waves in acoustic media).
 */
struct IsotropicAcousticConstantParams {
    real_t vp;          ///< P-wave velocity (m/s)
    real_t rho;         ///< Density (kg/m^3)
    real_t qkappa = 0;  ///< Q factor for bulk modulus (0 = no attenuation)
    // Note: No qmu for acoustic materials
};

// =============================================================================
// Grid Parameters (spatially-varying acoustic material from file)
// =============================================================================

// Note: Uses IsotropicAcousticMaterialData2D/3D from IsotropicAcousticIO.hpp

// =============================================================================
// ByAttribute Parameters (acoustic material per mesh attribute)
// =============================================================================

// Note: Uses IsotropicAcousticAttributeEntry from IsotropicAcousticIO.hpp

// =============================================================================
// ByAttributeMixed Parameters (mixed constant/grid per attribute)
// =============================================================================

/**
 * @struct IsotropicAcousticMixedAttributeEntry
 * @brief Entry for mixed attribute acoustic material (constant or grid per attribute)
 *
 * Used when config.format == "by_attribute_mixed".
 * Note: No vs or qmu for acoustic materials.
 */
struct IsotropicAcousticMixedAttributeEntry {
    int attribute;              ///< Mesh element attribute number
    /// Sub-mode: "constant" | "grid" | "adios2"
    std::string mode = "constant";
    /// Backward-compat: true iff mode == "grid" (set by Loader)
    bool is_heterogeneous = false;

    // Constant mode parameters
    real_t vp = 0.0;            ///< P-wave velocity (m/s)
    real_t rho = 0.0;           ///< Density (kg/m^3)
    real_t Qkappa = -1.0;       ///< Q factor for bulk modulus (-1 = no attenuation)
    // Note: No Qmu for acoustic materials

    // Grid mode parameters (loaded from ascii file)
    std::optional<IsotropicAcousticMaterialData2D> grid_data_2d;
    std::optional<IsotropicAcousticMaterialData3D> grid_data_3d;

    // ADIOS2 mode parameters (pre-computed GLL data from .bp file)
    std::shared_ptr<MaterialField> adios2_vp;
    std::shared_ptr<MaterialField> adios2_rho;
    std::shared_ptr<MaterialField> adios2_qkappa;  // visco-acoustic
};

// =============================================================================
// Combined Input Structure
// =============================================================================

/**
 * @struct IsotropicAcousticInput
 * @brief Unified input structure for creating IsotropicAcousticMaterial
 *
 * This structure holds all the information needed to create an
 * IsotropicAcousticMaterial, regardless of the data source.
 */
struct IsotropicAcousticInput {
    DataSource source;  ///< Which data source is used

    // Exactly one of these should be set based on `source`
    std::optional<IsotropicAcousticConstantParams> constant;                      ///< For DataSource::Constant
    std::optional<IsotropicAcousticMaterialData2D> grid_2d;              ///< For DataSource::Grid (2D)
    std::optional<IsotropicAcousticMaterialData3D> grid_3d;              ///< For DataSource::Grid (3D)
    std::optional<std::vector<IsotropicAcousticAttributeEntry>> by_attribute;  ///< For DataSource::ByAttribute
    std::optional<std::vector<IsotropicAcousticMixedAttributeEntry>> by_attribute_mixed;  ///< For DataSource::ByAttributeMixed

    // ADIOS2: pre-computed GLL data (vp, rho, optionally qkappa as MaterialField)
    std::shared_ptr<MaterialField> adios2_vp;      ///< For DataSource::ADIOS2
    std::shared_ptr<MaterialField> adios2_rho;     ///< For DataSource::ADIOS2
    std::shared_ptr<MaterialField> adios2_qkappa;  ///< For DataSource::ADIOS2 (visco-acoustic)

    // Attenuation settings (only qkappa, no qmu for acoustic)
    MaterialAttenuationConfig attenuation;

    // Dimension info
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
                return adios2_vp && adios2_rho;
        }
        return false;
    }

    /// Check if attenuation is enabled
    bool HasAttenuation() const {
        return attenuation.enabled;
    }
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ACOUSTIC_INPUT_HPP
