/**
 * @file IsotropicElasticLoader.hpp
 * @brief Loader for isotropic elastic material data
 *
 * Responsible for reading material data from various sources
 * and populating IsotropicElasticInput structure.
 *
 * Supported formats:
 * - Constant: Uniform properties from MaterialConfig
 * - Grid (ASCII): Spatially-varying data from file
 * - ByAttribute: Properties per mesh attribute from file
 *
 * Usage:
 *   MaterialConfig config = LoadMaterialConfig(yaml);
 *   IsotropicElasticInput data = IsotropicElasticLoader::Load(config, is_3d);
 *   auto material = IsotropicElasticBuilder<Dim>::Build(data, fes, ir);
 */

#ifndef SEM_ISOTROPIC_ELASTIC_LOADER_HPP
#define SEM_ISOTROPIC_ELASTIC_LOADER_HPP

#include "IsotropicElasticInput.hpp"
#include "config/ConfigTypes.hpp"
#include <string>

namespace SEM {

/**
 * @class IsotropicElasticLoader
 * @brief Loads isotropic elastic material data from configuration
 *
 * This class is responsible only for data loading, not material creation.
 * The loaded data is passed to IsotropicElasticBuilder for material creation.
 */
class IsotropicElasticLoader {
public:
    /**
     * @brief Load isotropic elastic material data from configuration
     *
     * @param config Material configuration (from YAML)
     * @param is_3d True for 3D simulation, false for 2D
     * @return IsotropicElasticInput Loaded data ready for Builder
     *
     * @throws MFEM_ABORT if file reading fails or format is unsupported
     */
    static IsotropicElasticInput Load(const MaterialConfig& config, bool is_3d);

private:
    /**
     * @brief Load constant material data
     * @param config Material configuration with vp, vs, rho values
     * @return IsotropicElasticInput with constant params
     */
    static IsotropicElasticInput LoadConstant(const MaterialConfig& config);

    /**
     * @brief Load grid material data from ASCII file (2D)
     * @param config Material configuration with file path
     * @return IsotropicElasticInput with 2D grid data
     */
    static IsotropicElasticInput LoadGrid2D(const MaterialConfig& config);

    /**
     * @brief Load grid material data from ASCII file (3D)
     * @param config Material configuration with file path
     * @return IsotropicElasticInput with 3D grid data
     */
    static IsotropicElasticInput LoadGrid3D(const MaterialConfig& config);

    /**
     * @brief Load by-attribute material data from file
     * @param config Material configuration with by_attribute file path
     * @return IsotropicElasticInput with attribute entries
     */
    static IsotropicElasticInput LoadByAttribute(const MaterialConfig& config);

    /**
     * @brief Load by-attribute-mixed material data
     *
     * Each attribute can have either constant values or grid-interpolated values.
     * Constant entries use params map, grid entries load from ASCII files.
     *
     * @param config Material configuration with attribute_entries already loaded
     * @param is_3d True for 3D simulation, false for 2D
     * @return IsotropicElasticInput with mixed attribute entries
     */
    static IsotropicElasticInput LoadByAttributeMixed(const MaterialConfig& config, bool is_3d);

    /**
     * @brief Load ADIOS2 (.bp) pre-computed GLL data
     *
     * Mirrors the acoustic version, but loads vp/vs/rho (and optionally
     * qkappa/qmu when attenuation is enabled).
     *
     * @param config Material configuration with adios2_vp/vs/rho/q*_file set
     * @param is_3d True for 3D simulation, false for 2D
     * @return IsotropicElasticInput with adios2 fields populated
     */
    static IsotropicElasticInput LoadADIOS2(const MaterialConfig& config, bool is_3d);

    /**
     * @brief Convert AttenuationConfig to MaterialAttenuationConfig
     * @param config AttenuationConfig from YAML
     * @return MaterialAttenuationConfig for IsotropicElasticInput
     */
    static MaterialAttenuationConfig ConvertAttenuation(const AttenuationConfig& config);
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ELASTIC_LOADER_HPP
