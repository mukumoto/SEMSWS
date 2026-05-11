/**
 * @file Visco_IsotropicAcousticSensitivity2D.hpp
 * @brief Sensitivity kernel for 2D visco-acoustic isotropic media
 *
 * Same κ and ρ kernels as the non-attenuating case.
 * The effect of attenuation is already included in the forward/adjoint wavefields.
 * Q gradient is not computed (future work).
 *
 * This is a typedef alias — visco-acoustic uses the same kernel class.
 */

#ifndef SEM_VISCO_ISOTROPIC_ACOUSTIC_SENSITIVITY_2D_HPP
#define SEM_VISCO_ISOTROPIC_ACOUSTIC_SENSITIVITY_2D_HPP

#include "fwi/IsotropicAcousticSensitivity2D.hpp"

namespace SEM {

using Visco_IsotropicAcousticSensitivity2D = IsotropicAcousticSensitivity2D;

}  // namespace SEM

#endif  // SEM_VISCO_ISOTROPIC_ACOUSTIC_SENSITIVITY_2D_HPP
