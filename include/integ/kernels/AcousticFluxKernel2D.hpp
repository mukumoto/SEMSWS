/**
 * @file AcousticFluxKernel2D.hpp
 * @brief Element-local quadrature-point flux for isotropic acoustic 2D.
 *
 * Pure function templated on the scalar type T. The material coefficient
 * (inv_rho = 1/ρ) is passed as T so callers can pick:
 *   - T = real_t                          → forward stiffness action (K·φ)
 *   - T = mfem::future::dual<real_t,T2>   → forward-mode AD sensitivity
 *
 * The wavefield gradient components (dp_xi, dp_eta) and the geometry
 * (inverse Jacobian entries j0..j3 plus weighted Jacobian determinant w)
 * stay real_t — they are passive inputs for the AD caller and a minor
 * storage win for the forward caller.
 *
 * Math (shared between both callers):
 *     dp_dx = dp_xi * j0 + dp_eta * j3    (physical gradient from reference)
 *     dp_dy = dp_xi * j2 + dp_eta * j1
 *     qx    = inv_rho * dp_dx             ← ONLY line that depends on m
 *     qy    = inv_rho * dp_dy
 *     out_flux_xi  = (qx * j0 + qy * j2) * w   (ref-frame flux component)
 *     out_flux_eta = (qx * j3 + qy * j1) * w
 *
 * References:
 *   - Pre-refactor body: IsotropicAcousticKernels2D.cpp L119-L130 (identical math).
 *   - MFEM dual primitive: /home/kota/program_ubuntu/mfem/linalg/dual.hpp.
 */

#ifndef SEM_ACOUSTIC_FLUX_KERNEL_2D_HPP
#define SEM_ACOUSTIC_FLUX_KERNEL_2D_HPP

#include <mfem.hpp>

namespace SEM {

using mfem::real_t;

/**
 * @brief Physical-frame flux: q = (1/ρ) ∇p at one quadrature point.
 *
 * This is the minimal material-dependent building block. The output is
 * the physical-frame flux with NO Jacobian transform and NO quadrature
 * weight applied — callers handle those based on what they need.
 *
 * Used by:
 *   - AcousticFlux2D() below (forward stiffness action, adds transform + weight)
 *   - AD sensitivity kernel (contracts directly with adjoint physical gradient)
 */
template <typename T>
MFEM_HOST_DEVICE inline
void AcousticFluxPhysical2D(const T& inv_rho_val,
                            real_t dp_dx, real_t dp_dy,
                            T& qx, T& qy)
{
    qx = inv_rho_val * dp_dx;
    qy = inv_rho_val * dp_dy;
}

/**
 * @brief Weak-form reference-frame flux for the forward stiffness kernel.
 *
 * Composes AcousticFluxPhysical2D with the reference↔physical gradient
 * transform and the quadrature weight. Convenience wrapper used by
 * the forward stiffness action (K·φ). Sensitivity code normally calls
 * AcousticFluxPhysical2D directly.
 */
template <typename T>
MFEM_HOST_DEVICE inline
void AcousticFlux2D(const T& inv_rho_val,
                    real_t dp_xi, real_t dp_eta,
                    real_t j0, real_t j1, real_t j2, real_t j3, real_t w,
                    T& out_flux_xi, T& out_flux_eta)
{
    // Reference → physical gradient (geometry scalars are passive).
    const real_t dp_dx = dp_xi * j0 + dp_eta * j3;
    const real_t dp_dy = dp_xi * j2 + dp_eta * j1;

    T qx, qy;
    AcousticFluxPhysical2D(inv_rho_val, dp_dx, dp_dy, qx, qy);

    // Transform back to reference and apply quadrature weight.
    out_flux_xi  = (qx * j0 + qy * j2) * w;
    out_flux_eta = (qx * j3 + qy * j1) * w;
}

}  // namespace SEM

#endif  // SEM_ACOUSTIC_FLUX_KERNEL_2D_HPP
