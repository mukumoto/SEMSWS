/**
 * @file Visco_IsotropicAcousticSensitivityAD3D.hpp
 * @brief AD-based sensitivity kernel for 3D visco-acoustic media (non-Q path).
 *
 * 3D counterpart to Visco_IsotropicAcousticSensitivityAD2D. Typedef alias
 * to the pure AD class — the K_Vp and K_ρ formulas are identical to pure
 * because memory variables are passive wrt the current step's material.
 * Q inversion requires the separate `..._Q` class (Phase V1).
 */

#ifndef SEM_VISCO_ISOTROPIC_ACOUSTIC_SENSITIVITY_AD_3D_HPP
#define SEM_VISCO_ISOTROPIC_ACOUSTIC_SENSITIVITY_AD_3D_HPP

#include "fwi/IsotropicAcousticSensitivityAD3D.hpp"

namespace SEM {

using Visco_IsotropicAcousticSensitivityAD3D = IsotropicAcousticSensitivityAD3D;

}  // namespace SEM

#endif  // SEM_VISCO_ISOTROPIC_ACOUSTIC_SENSITIVITY_AD_3D_HPP
