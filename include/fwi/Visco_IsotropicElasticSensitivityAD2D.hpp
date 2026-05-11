/**
 * @file Visco_IsotropicElasticSensitivityAD2D.hpp
 * @brief AD-based sensitivity kernel for 2D visco-elastic media (non-Q path).
 *
 * Typedef alias to pure elastic AD. σ(x,t) = σ_elastic(λ,μ,ε(u)) − Σ_l r_l,
 * so at each instant ∂σ/∂λ and ∂σ/∂μ reduce to their elastic forms; the
 * memory correction only affects the wavefields u, λ* through the visco
 * time-stepping. K_Vp / K_Vs / K_ρ formulas are therefore identical to pure.
 *
 * Q inversion: separate `..._Q` class (Phase V2).
 */

#ifndef SEM_VISCO_ISOTROPIC_ELASTIC_SENSITIVITY_AD_2D_HPP
#define SEM_VISCO_ISOTROPIC_ELASTIC_SENSITIVITY_AD_2D_HPP

#include "fwi/IsotropicElasticSensitivityAD2D.hpp"

namespace SEM {

using Visco_IsotropicElasticSensitivityAD2D = IsotropicElasticSensitivityAD2D;

}  // namespace SEM

#endif  // SEM_VISCO_ISOTROPIC_ELASTIC_SENSITIVITY_AD_2D_HPP
