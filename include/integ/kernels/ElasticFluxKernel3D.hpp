/**
 * @file ElasticFluxKernel3D.hpp
 * @brief 3D isotropic elastic stress + weak-form flux template.
 *
 * 3D counterpart to ElasticFluxKernel2D.hpp. Templated on T so the same
 * stress formula serves forward (T = real_t) and forward-mode AD sensitivity
 * (T = mfem::future::dual<real_t, real_t>).
 *
 * Stress (isotropic, plane-strain-free 3D):
 *   trace  = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z
 *   σxx = λ·trace + 2μ·∂ux/∂x
 *   σyy = λ·trace + 2μ·∂uy/∂y
 *   σzz = λ·trace + 2μ·∂uz/∂z
 *   σxy = μ·(∂ux/∂y + ∂uy/∂x)
 *   σxz = μ·(∂ux/∂z + ∂uz/∂x)
 *   σyz = μ·(∂uy/∂z + ∂uz/∂y)
 */

#ifndef SEM_ELASTIC_FLUX_KERNEL_3D_HPP
#define SEM_ELASTIC_FLUX_KERNEL_3D_HPP

#include <mfem.hpp>

namespace SEM {

using mfem::real_t;

/// Physical-frame 3D stress (6 components σxx, σyy, σzz, σxy, σxz, σyz).
/// No Jacobian transform, no quadrature weight.
template <typename T>
MFEM_HOST_DEVICE inline
void ElasticStressPhysical3D(const T& lambda_val, const T& mu_val,
                              real_t dux_dx, real_t dux_dy, real_t dux_dz,
                              real_t duy_dx, real_t duy_dy, real_t duy_dz,
                              real_t duz_dx, real_t duz_dy, real_t duz_dz,
                              T& sigma_xx, T& sigma_yy, T& sigma_zz,
                              T& sigma_xy, T& sigma_xz, T& sigma_yz)
{
    const real_t trace = dux_dx + duy_dy + duz_dz;
    const T two_mu = mu_val + mu_val;
    sigma_xx = lambda_val * trace + two_mu * dux_dx;
    sigma_yy = lambda_val * trace + two_mu * duy_dy;
    sigma_zz = lambda_val * trace + two_mu * duz_dz;
    sigma_xy = mu_val * (dux_dy + duy_dx);
    sigma_xz = mu_val * (dux_dz + duz_dx);
    sigma_yz = mu_val * (duy_dz + duz_dy);
}

}  // namespace SEM

#endif  // SEM_ELASTIC_FLUX_KERNEL_3D_HPP
