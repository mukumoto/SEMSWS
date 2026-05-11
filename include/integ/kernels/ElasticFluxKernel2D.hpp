/**
 * @file ElasticFluxKernel2D.hpp
 * @brief Element-local stress + reference-frame flux for isotropic
 *        elastic 2D (plane strain).
 *
 * Pure function template on a scalar type T that carries the Lamé
 * parameters (λ, μ). Callers pick:
 *   - T = real_t                              → forward stiffness action
 *   - T = mfem::future::dual<real_t, real_t>  → forward-mode AD sensitivity
 *
 * Geometry scalars (j0..j3, w) and displacement gradient entries stay
 * real_t — they are passive inputs with respect to the material seed.
 *
 * Math mirrors the existing hand kernel IsotropicElasticKernels2D.cpp:
 *     trace  = ∂ux/∂x + ∂uy/∂y       = ∇·u
 *     σxx    = λ·trace + 2μ ∂ux/∂x
 *     σyy    = λ·trace + 2μ ∂uy/∂y
 *     σxy    =   μ (∂ux/∂y + ∂uy/∂x)
 *     out_flux_x_xi  = (σxx j0 + σxy j2) w
 *     out_flux_y_xi  = (σxy j0 + σyy j2) w
 *     out_flux_x_eta = (σxx j3 + σxy j1) w
 *     out_flux_y_eta = (σxy j3 + σyy j1) w
 *
 * Components are labelled (fx, fy) for the displacement axis and (xi, eta)
 * for the reference derivative they pair with in the weak form.
 *
 * A single scalar type T applies to BOTH λ and μ so a single-seed
 * `mfem::future::dual<real_t, real_t>` caller can zero the off-path
 * gradient and get the correct tangent for whichever material is being
 * perturbed.
 */

#ifndef SEM_ELASTIC_FLUX_KERNEL_2D_HPP
#define SEM_ELASTIC_FLUX_KERNEL_2D_HPP

#include <mfem.hpp>

namespace SEM {

using mfem::real_t;

/// Physical-frame stress components (symmetric 2×2: σxx, σxy, σyy).
/// No Jacobian transform, no quadrature weight applied here; the caller
/// decides whether it needs the weak-form flux (ElasticFlux2D) or the
/// raw stress (e.g. contracted directly against an adjoint strain).
template <typename T>
MFEM_HOST_DEVICE inline
void ElasticStressPhysical2D(const T& lambda_val, const T& mu_val,
                              real_t dux_dx, real_t dux_dy,
                              real_t duy_dx, real_t duy_dy,
                              T& sigma_xx, T& sigma_xy, T& sigma_yy)
{
    const real_t trace = dux_dx + duy_dy;
    const T two_mu = mu_val + mu_val;
    sigma_xx = lambda_val * trace + two_mu * dux_dx;
    sigma_yy = lambda_val * trace + two_mu * duy_dy;
    sigma_xy = mu_val * (dux_dy + duy_dx);
}

/// Weak-form reference-frame flux for the forward stiffness kernel.
/// Composes stress computation with the reference↔physical gradient
/// transform and the quadrature weight. Four outputs: the per-component
/// contribution (x, y) to each reference derivative direction (ξ, η).
template <typename T>
MFEM_HOST_DEVICE inline
void ElasticFlux2D(const T& lambda_val, const T& mu_val,
                   real_t dux_xi, real_t dux_eta,
                   real_t duy_xi, real_t duy_eta,
                   real_t j0, real_t j1, real_t j2, real_t j3, real_t w,
                   T& out_flux_x_xi, T& out_flux_y_xi,
                   T& out_flux_x_eta, T& out_flux_y_eta)
{
    // Reference → physical displacement gradient (geometry is passive).
    const real_t dux_dx = dux_xi * j0 + dux_eta * j3;
    const real_t dux_dy = dux_xi * j2 + dux_eta * j1;
    const real_t duy_dx = duy_xi * j0 + duy_eta * j3;
    const real_t duy_dy = duy_xi * j2 + duy_eta * j1;

    T sigma_xx, sigma_xy, sigma_yy;
    ElasticStressPhysical2D(lambda_val, mu_val,
                            dux_dx, dux_dy, duy_dx, duy_dy,
                            sigma_xx, sigma_xy, sigma_yy);

    // Transform back to reference and apply quadrature weight.
    out_flux_x_xi  = (sigma_xx * j0 + sigma_xy * j2) * w;
    out_flux_y_xi  = (sigma_xy * j0 + sigma_yy * j2) * w;
    out_flux_x_eta = (sigma_xx * j3 + sigma_xy * j1) * w;
    out_flux_y_eta = (sigma_xy * j3 + sigma_yy * j1) * w;
}

}  // namespace SEM

#endif  // SEM_ELASTIC_FLUX_KERNEL_2D_HPP
