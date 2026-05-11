/**
 * @file IsotropicAcousticLoader.hpp
 * @brief Loader for isotropic acoustic material data
 *
 * Responsible for reading acoustic material data from various sources
 * and populating IsotropicAcousticInput structure.
 *
 * Note: Acoustic materials have no shear wave (vs) and no Qmu.
 */

#ifndef SEM_ISOTROPIC_ACOUSTIC_LOADER_HPP
#define SEM_ISOTROPIC_ACOUSTIC_LOADER_HPP

#include "IsotropicAcousticInput.hpp"
#include "config/ConfigTypes.hpp"
#include <string>

namespace SEM {

/**
 * @class IsotropicAcousticLoader
 * @brief Loads isotropic acoustic material data from configuration
 */
class IsotropicAcousticLoader {
public:
    /**
     * @brief Load isotropic acoustic material data from configuration
     *
     * @param config Material configuration (from YAML)
     * @param is_3d True for 3D simulation, false for 2D
     * @return IsotropicAcousticInput Loaded data ready for Builder
     */
    static IsotropicAcousticInput Load(const MaterialConfig& config, bool is_3d);

private:
    static IsotropicAcousticInput LoadConstant(const MaterialConfig& config);
    static IsotropicAcousticInput LoadGrid2D(const MaterialConfig& config);
    static IsotropicAcousticInput LoadGrid3D(const MaterialConfig& config);
    static IsotropicAcousticInput LoadByAttribute(const MaterialConfig& config);
    static IsotropicAcousticInput LoadByAttributeMixed(const MaterialConfig& config, bool is_3d);
    static IsotropicAcousticInput LoadADIOS2(const MaterialConfig& config, bool is_3d);
    static MaterialAttenuationConfig ConvertAttenuation(const AttenuationConfig& config);
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ACOUSTIC_LOADER_HPP
