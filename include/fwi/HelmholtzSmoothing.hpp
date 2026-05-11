/**
 * @file HelmholtzSmoothing.hpp
 * @brief Helmholtz smoothing for sensitivity kernels
 *
 * Solves (M + σ²K) u_s = M u via AMG-preconditioned CG.
 *
 * Variable σ(x) = α · Vp(x) / f:
 *   σ(x)² = (α/f)² · Vp(x)²
 *   K_σ² = ∫ σ(x)² ∇φ·∇ψ dx  (DiffusionIntegrator with Vp² coefficient)
 *
 * Constant σ (simpler case):
 *   K_σ² = σ² · ∫ ∇φ·∇ψ dx  (standard DiffusionIntegrator)
 *
 * restrict_attrs: if non-empty, M and K are assembled only on elements
 * with attribute in the set. DOFs outside the restricted region keep
 * their input values unchanged.
 */

#ifndef SEM_HELMHOLTZ_SMOOTHING_HPP
#define SEM_HELMHOLTZ_SMOOTHING_HPP

#include <mfem.hpp>
#include <vector>

namespace SEM {

using namespace mfem;

void HelmholtzSmooth(ParFiniteElementSpace& fes,
                     ParGridFunction& gf,
                     const ParGridFunction& vp_gf,
                     real_t alpha, real_t freq,
                     const std::vector<int>& restrict_attrs = {});

void HelmholtzSmooth(ParFiniteElementSpace& fes,
                     ParGridFunction& gf,
                     real_t sigma,
                     const std::vector<int>& restrict_attrs = {});

void HelmholtzSmoothAniso(ParFiniteElementSpace& fes,
                           ParGridFunction& gf,
                           const ParGridFunction& vp_gf,
                           real_t alpha_x, real_t alpha_y,
                           real_t freq,
                           const std::vector<int>& restrict_attrs = {});

void HelmholtzSmoothAniso(ParFiniteElementSpace& fes,
                           ParGridFunction& gf,
                           real_t sigma_x, real_t sigma_y,
                           const std::vector<int>& restrict_attrs = {});

}  // namespace SEM

#endif  // SEM_HELMHOLTZ_SMOOTHING_HPP
