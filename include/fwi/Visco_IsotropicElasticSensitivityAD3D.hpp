/**
 * @file Visco_IsotropicElasticSensitivityAD3D.hpp
 * @brief AD-based sensitivity kernel for 3D visco-elastic media (non-Q path).
 *
 * 3D counterpart to Visco_IsotropicElasticSensitivityAD2D. Typedef alias
 * to pure 3D elastic AD. Memory variables are passive wrt the current step's
 * material so K_Vp/K_Vs/K_ρ are identical to pure.
 */

#ifndef SEM_VISCO_ISOTROPIC_ELASTIC_SENSITIVITY_AD_3D_HPP
#define SEM_VISCO_ISOTROPIC_ELASTIC_SENSITIVITY_AD_3D_HPP

#include "fwi/IsotropicElasticSensitivityAD3D.hpp"

namespace SEM {

using Visco_IsotropicElasticSensitivityAD3D = IsotropicElasticSensitivityAD3D;

}  // namespace SEM

#endif  // SEM_VISCO_ISOTROPIC_ELASTIC_SENSITIVITY_AD_3D_HPP
