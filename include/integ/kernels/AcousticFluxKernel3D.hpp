/**
 * @file AcousticFluxKernel3D.hpp
 * @brief Element-local quadrature-point flux for isotropic acoustic 3D.
 *
 * 3D counterpart to AcousticFluxKernel2D.hpp. Templated on a scalar type T
 * so the same routine serves forward (T = real_t) and forward-mode AD
 * sensitivity (T = mfem::future::dual<real_t, real_t>).
 *
 * Forward math (same as IsotropicAcousticKernels3D.cpp flux block):
 *   qx = inv_rho · dp/dx
 *   qy = inv_rho · dp/dy
 *   qz = inv_rho · dp/dz
 *   out_flux_xi   = (qx·j0 + qy·j1 + qz·j2) · w
 *   out_flux_eta  = (qx·j3 + qy·j4 + qz·j5) · w
 *   out_flux_zeta = (qx·j6 + qy·j7 + qz·j8) · w
 * where jK are row-major entries of invJ and w = weight · detJ.
 */

#ifndef SEM_ACOUSTIC_FLUX_KERNEL_3D_HPP
#define SEM_ACOUSTIC_FLUX_KERNEL_3D_HPP

#include <mfem.hpp>

namespace SEM {

using mfem::real_t;

/// Physical-frame flux q = (1/ρ) ∇p. No Jacobian transform, no weight.
template <typename T>
MFEM_HOST_DEVICE inline
void AcousticFluxPhysical3D(const T& inv_rho_val,
                             real_t dp_dx, real_t dp_dy, real_t dp_dz,
                             T& qx, T& qy, T& qz)
{
    qx = inv_rho_val * dp_dx;
    qy = inv_rho_val * dp_dy;
    qz = inv_rho_val * dp_dz;
}

/// Weak-form reference-frame flux: physical flux transformed back to (ξ,η,ζ)
/// and multiplied by the quadrature weight. Geometry scalars are passive
/// wrt the material seed.
template <typename T>
MFEM_HOST_DEVICE inline
void AcousticFlux3D(const T& inv_rho_val,
                    real_t dp_xi, real_t dp_eta, real_t dp_zeta,
                    real_t j0, real_t j1, real_t j2,
                    real_t j3, real_t j4, real_t j5,
                    real_t j6, real_t j7, real_t j8,
                    real_t w,
                    T& out_flux_xi, T& out_flux_eta, T& out_flux_zeta)
{
    const real_t dp_dx = dp_xi * j0 + dp_eta * j3 + dp_zeta * j6;
    const real_t dp_dy = dp_xi * j1 + dp_eta * j4 + dp_zeta * j7;
    const real_t dp_dz = dp_xi * j2 + dp_eta * j5 + dp_zeta * j8;

    T qx, qy, qz;
    AcousticFluxPhysical3D(inv_rho_val, dp_dx, dp_dy, dp_dz, qx, qy, qz);

    out_flux_xi   = (qx * j0 + qy * j1 + qz * j2) * w;
    out_flux_eta  = (qx * j3 + qy * j4 + qz * j5) * w;
    out_flux_zeta = (qx * j6 + qy * j7 + qz * j8) * w;
}

}  // namespace SEM

#endif  // SEM_ACOUSTIC_FLUX_KERNEL_3D_HPP
