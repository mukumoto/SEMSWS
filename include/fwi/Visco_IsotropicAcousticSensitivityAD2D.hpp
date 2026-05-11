/**
 * @file Visco_IsotropicAcousticSensitivityAD2D.hpp
 * @brief AD-based sensitivity kernel for 2D visco-acoustic media (non-Q path).
 *
 * Typedef alias to the pure-acoustic AD class. At each time step the
 * per-point partials ∂σ/∂λ etc. pick up only the elastic part: the memory
 * variables {r_l} depend on the ε history, not on the current step's
 * material, so they're passive wrt λ/μ/ρ at this instant. The accumulated
 * history effect is carried by the forward/adjoint wavefields themselves
 * (attenuated by the visco time-stepping).
 *
 * Limitations:
 * - Q (Qκ) is NOT an inversion parameter here. To invert for Qκ, use the
 *   upcoming `Visco_IsotropicAcousticSensitivityAD2D_Q` (Phase V1).
 * - Correctness depends on the adjoint viscoelastic equation being
 *   implemented; see `AdjointSimulation::AdjointStep`.
 */

#ifndef SEM_VISCO_ISOTROPIC_ACOUSTIC_SENSITIVITY_AD_2D_HPP
#define SEM_VISCO_ISOTROPIC_ACOUSTIC_SENSITIVITY_AD_2D_HPP

#include "fwi/IsotropicAcousticSensitivityAD2D.hpp"

namespace SEM {

using Visco_IsotropicAcousticSensitivityAD2D = IsotropicAcousticSensitivityAD2D;

}  // namespace SEM

#endif  // SEM_VISCO_ISOTROPIC_ACOUSTIC_SENSITIVITY_AD_2D_HPP
