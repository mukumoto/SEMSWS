/**
 * @file Visco_IsotropicElasticKernels3D.cpp
 * @brief 3D viscoelastic (isotropic elastic with attenuation) stiffness kernel implementations
 *
 * GPU-optimized implementation using forall_3D + shared memory.
 * Each element gets NGLL³ threads for parallel computation.
 *
 * Memory layout: (ix, iy, iz, element) - Element is SLOWEST (matches libCEED/MFEM).
 * Memory variables (M1-M6) are updated per-GLL-point within the kernel.
 */

#include "integ/SEMVisco_IsotropicElasticIntegrator.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "common/GpuMacros.hpp"
#include <mfem.hpp>

namespace SEM {

// =============================================================================
// 3D Viscoelastic Kernel (GPU-Optimized: forall_3D + Shared Memory)
// =============================================================================

template<int NGLL>
void SEMVisco_IsotropicElasticIntegrator3D::AddMultPA_Visco_Opt(const Vector &u, Vector &f) const
{
    const int ne = ne_;


    const real_t* d_u = u.Read();
    real_t* d_f = f.ReadWrite();

    // DOF ordering views via factory: gather_map_x(ix, iy, iz, e)
    auto gather_map_x = dofs_.ViewGatherMapX();
    auto gather_map_y = dofs_.ViewGatherMapY();
    auto gather_map_z = dofs_.ViewGatherMapZ();

    // Material params view: params(comp, ix, iy, iz, e) where comp: 0=kappa, 1=mu
    auto params = ViewMaterialParams();

    // Packed inverse Jacobian + detJ view: invJPacked(comp, ix, iy, iz, e)
    // Components: 0-8 = invJ (row-major), 9 = detJ
    // This layout provides coalesced memory access for all 10 values per point
    auto invJPacked = geom_.ViewInvJPacked();

    // Shape derivative views: dshape(k, ix) = ℓ'_k(ξ_ix)
    //   = derivative of the k-th 1D Lagrange basis evaluated at the ix-th GLL node.
    // Data stored row-major in geom_.dxshape[ngll * ngll]:
    //   dxshape[i * ngll + j] = ℓ'_j(ξ_i)
    //   (from SEMGeometry: hex_fe.CalcDShape at IntPoint(i) with lex[j]=(j,0,0);
    //    tensor-product basis makes dshape_mat(lex[j], 0) reduce to ℓ'_j(ξ_i))
    // MFEM Reshape is column-major: view(a, b) = data[a + b * ngll]
    //   → dshape(k, ix)   = data[ix * ngll + k] = ℓ'_k(ξ_ix)   (Phase 1 gradient)
    //   → dshape_w(ix, k) = data[k * ngll + ix] = ℓ'_ix(ξ_k)·w_k (Phase 2 scatter)
    // Same 1D basis for ξ/η/ζ directions (tensor product element).
    //
    // dshape_w is quadrature-weighted:
    //   dxshape_w[i * ngll + j] = ℓ'_j(ξ_i) · w_i   (w_i = GLL weight at node i)
    auto dshape = geom_.ViewDShape();
    auto dshape_w = geom_.ViewDShapeW();

    // Quadrature weight view (same for all directions in tensor product element)
    auto wgll = geom_.ViewWgll();

    // Get memory state from attenuation model
    ViscoelasticMemory3D* memory = const_cast<GeneralizedZener3D&>(attenuation_).GetMemory();
    MFEM_VERIFY(memory != nullptr, "Viscoelastic memory not initialized");

    const int n_units = memory->NumUnits();

    // Memory variable View [6, ngllx, nglly, ngllz, ne, n_units]
    auto M = memory->ViewMWrite();

    // Old strain View [6, ngllx, nglly, ngllz, ne]
    auto strain_old = memory->ViewStrainOldWrite();

    // Crank-Nicolson coefficient View [4, ngllx, nglly, ngllz, ne, n_units]
    auto coeffs = memory->ViewCoeffs();

    // GPU-optimized element loop: NGLL³ threads per element
    // Split into multiple subphases to reduce register pressure (target: <60 REG)
    // Standard SEM weak-form computation 
    mfem::forall_3D(ne, NGLL, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei)
    {
        // ===== Shared memory for element data =====
        // On GPU: __shared__ memory, fast access within thread-block
        // On CPU: Just local arrays (stack or thread-private)
        MFEM_SHARED real_t s_disp_x[NGLL][NGLL][NGLL];
        MFEM_SHARED real_t s_disp_y[NGLL][NGLL][NGLL];
        MFEM_SHARED real_t s_disp_z[NGLL][NGLL][NGLL];
        MFEM_SHARED real_t s_hprime[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime_w[NGLL][NGLL];

        // Stress coefficients (9 components for 3D weak form)
        MFEM_SHARED real_t s_coeff[9][NGLL][NGLL][NGLL];

        // ===== Phase 1: Load basis functions to shared memory =====
        if (MFEM_THREAD_ID(z) == 0)
        {
            MFEM_FOREACH_THREAD(iy, y, NGLL)
            {
                MFEM_FOREACH_THREAD(ix, x, NGLL)
                {
                    s_hprime[ix][iy] = dshape(iy, ix);
                    s_hprime_w[ix][iy] = dshape_w(iy, ix);
                }
            }
        }

        // ===== Phase 2: Gather displacement from global to shared =====
        MFEM_FOREACH_THREAD(iz, z, NGLL)
        {
            MFEM_FOREACH_THREAD(iy, y, NGLL)
            {
                MFEM_FOREACH_THREAD(ix, x, NGLL)
                {
                    s_disp_x[iz][iy][ix] = d_u[gather_map_x(ix, iy, iz, ei)];
                    s_disp_y[iz][iy][ix] = d_u[gather_map_y(ix, iy, iz, ei)];
                    s_disp_z[iz][iy][ix] = d_u[gather_map_z(ix, iy, iz, ei)];

                }
            }
        }
        MFEM_SYNC_THREAD;

        // ===== Phase 3: Compute strain and stress =====
        // Only read M values to subtract from stress, do NOT update M here
        // M update is deferred to Phase 5 (after scatter) to reduce register pressure
        MFEM_FOREACH_THREAD(iz, z, NGLL)
        {
            MFEM_FOREACH_THREAD(iy, y, NGLL)
            {
                MFEM_FOREACH_THREAD(ix, x, NGLL)
                {
                    // Compute reference gradients via sum factorization
                    real_t tmpx_x = 0, tmpx_y = 0, tmpx_z = 0;
                    real_t tmpy_x = 0, tmpy_y = 0, tmpy_z = 0;
                    real_t tmpz_x = 0, tmpz_y = 0, tmpz_z = 0;

                    SEM_NOUNROLL
                    for (int k = 0; k < NGLL; k++)
                    {
                        const real_t hp_x = s_hprime[ix][k];
                        tmpx_x += s_disp_x[iz][iy][k] * hp_x;
                        tmpy_x += s_disp_y[iz][iy][k] * hp_x;
                        tmpz_x += s_disp_z[iz][iy][k] * hp_x;

                        const real_t hp_y = s_hprime[iy][k];
                        tmpx_y += s_disp_x[iz][k][ix] * hp_y;
                        tmpy_y += s_disp_y[iz][k][ix] * hp_y;
                        tmpz_y += s_disp_z[iz][k][ix] * hp_y;

                        const real_t hp_z = s_hprime[iz][k];
                        tmpx_z += s_disp_x[k][iy][ix] * hp_z;
                        tmpy_z += s_disp_y[k][iy][ix] * hp_z;
                        tmpz_z += s_disp_z[k][iy][ix] * hp_z;
                    }

                    // Load inverse Jacobian from packed layout
                    const real_t j0 = invJPacked(0, ix, iy, iz, ei);
                    const real_t j1 = invJPacked(1, ix, iy, iz, ei);
                    const real_t j2 = invJPacked(2, ix, iy, iz, ei);
                    const real_t j3 = invJPacked(3, ix, iy, iz, ei);
                    const real_t j4 = invJPacked(4, ix, iy, iz, ei);
                    const real_t j5 = invJPacked(5, ix, iy, iz, ei);
                    const real_t j6 = invJPacked(6, ix, iy, iz, ei);
                    const real_t j7 = invJPacked(7, ix, iy, iz, ei);
                    const real_t j8 = invJPacked(8, ix, iy, iz, ei);

                    // Compute strain components
                    const real_t ekk = (tmpx_x * j0 + tmpx_y * j3 + tmpx_z * j6 +
                                        tmpy_x * j1 + tmpy_y * j4 + tmpy_z * j7 +
                                        tmpz_x * j2 + tmpz_y * j5 + tmpz_z * j8);

                    const real_t d11 = (tmpx_x * j0 + tmpx_y * j3 + tmpx_z * j6) - ekk / 3.0;
                    const real_t d22 = (tmpy_x * j1 + tmpy_y * j4 + tmpy_z * j7) - ekk / 3.0;
                    const real_t d12 = 0.5 * (tmpx_x * j1 + tmpx_y * j4 + tmpx_z * j7 +
                                              tmpy_x * j0 + tmpy_y * j3 + tmpy_z * j6);
                    const real_t d13 = 0.5 * (tmpx_x * j2 + tmpx_y * j5 + tmpx_z * j8 +
                                              tmpz_x * j0 + tmpz_y * j3 + tmpz_z * j6);
                    const real_t d23 = 0.5 * (tmpy_x * j2 + tmpy_y * j5 + tmpy_z * j8 +
                                              tmpz_x * j1 + tmpz_y * j4 + tmpz_z * j7);

                    // Load material parameters
                    const real_t kappa = params(0, ix, iy, iz, ei);
                    const real_t mu_val = params(1, ix, iy, iz, ei);

                    // Elastic stress using UNRELAXED moduli
                    real_t sigma_xx = kappa * ekk + 2.0 * mu_val * d11;
                    real_t sigma_yy = kappa * ekk + 2.0 * mu_val * d22;
                    real_t sigma_zz = kappa * ekk - 2.0 * mu_val * (d11 + d22);
                    real_t sigma_xy = 2.0 * mu_val * d12;
                    real_t sigma_xz = 2.0 * mu_val * d13;
                    real_t sigma_yz = 2.0 * mu_val * d23;

                    // Sum M contributions (READ ONLY - no update here)
                    // Using packed layout: M(comp, ix, iy, iz, e, unit) for coalesced access
                    SEM_NOUNROLL
                    for (int uu = 0; uu < n_units; uu++)
                    {
                        // Read current M values
                        const real_t m1 = M(0, ix, iy, iz, ei, uu);
                        const real_t m2 = M(1, ix, iy, iz, ei, uu);
                        const real_t m3 = M(2, ix, iy, iz, ei, uu);
                        const real_t m4 = M(3, ix, iy, iz, ei, uu);
                        const real_t m5 = M(4, ix, iy, iz, ei, uu);
                        const real_t m6 = M(5, ix, iy, iz, ei, uu);

                        // Subtract memory variable contribution from stress
                        sigma_xx -= kappa * m1 + mu_val * m2;
                        sigma_yy -= kappa * m1 + mu_val * m3;
                        sigma_zz -= kappa * m1 - mu_val * (m2 + m3);
                        sigma_xy -= mu_val * m4;
                        sigma_xz -= mu_val * m5;
                        sigma_yz -= mu_val * m6;
                    }

                    // Weak form back-transformation: sigma · J^{-T} * detJ
                    const real_t w = invJPacked(9, ix, iy, iz, ei);

                    s_coeff[0][iz][iy][ix] = (sigma_xx * j0 + sigma_xy * j1 + sigma_xz * j2) * w;
                    s_coeff[1][iz][iy][ix] = (sigma_xy * j3 + sigma_yy * j4 + sigma_yz * j5) * w;
                    s_coeff[2][iz][iy][ix] = (sigma_xz * j6 + sigma_yz * j7 + sigma_zz * j8) * w;
                    s_coeff[3][iz][iy][ix] = (sigma_xx * j3 + sigma_xy * j4 + sigma_xz * j5) * w;
                    s_coeff[4][iz][iy][ix] = (sigma_xx * j6 + sigma_xy * j7 + sigma_xz * j8) * w;
                    s_coeff[5][iz][iy][ix] = (sigma_xy * j0 + sigma_yy * j1 + sigma_yz * j2) * w;
                    s_coeff[6][iz][iy][ix] = (sigma_xy * j6 + sigma_yy * j7 + sigma_yz * j8) * w;
                    s_coeff[7][iz][iy][ix] = (sigma_xz * j0 + sigma_yz * j1 + sigma_zz * j2) * w;
                    s_coeff[8][iz][iy][ix] = (sigma_xz * j3 + sigma_yz * j4 + sigma_zz * j5) * w;
                }
            }
        }
        MFEM_SYNC_THREAD;

        // ===== Phase 4: Compute force and scatter to global =====
        MFEM_FOREACH_THREAD(iz, z, NGLL)
        {
            MFEM_FOREACH_THREAD(iy, y, NGLL)
            {
                MFEM_FOREACH_THREAD(ix, x, NGLL)
                {
                    real_t tmpx_x = 0, tmpx_y = 0, tmpx_z = 0;
                    real_t tmpy_x = 0, tmpy_y = 0, tmpy_z = 0;
                    real_t tmpz_x = 0, tmpz_y = 0, tmpz_z = 0;

                    SEM_NOUNROLL
                    for (int k = 0; k < NGLL; k++)
                    {
                        const real_t hpw_x = s_hprime_w[k][ix];
                        tmpx_x += s_coeff[0][iz][iy][k] * hpw_x;
                        tmpy_x += s_coeff[5][iz][iy][k] * hpw_x;
                        tmpz_x += s_coeff[7][iz][iy][k] * hpw_x;

                        const real_t hpw_y = s_hprime_w[k][iy];
                        tmpx_y += s_coeff[3][iz][k][ix] * hpw_y;
                        tmpy_y += s_coeff[1][iz][k][ix] * hpw_y;
                        tmpz_y += s_coeff[8][iz][k][ix] * hpw_y;

                        const real_t hpw_z = s_hprime_w[k][iz];
                        tmpx_z += s_coeff[4][k][iy][ix] * hpw_z;
                        tmpy_z += s_coeff[6][k][iy][ix] * hpw_z;
                        tmpz_z += s_coeff[2][k][iy][ix] * hpw_z;
                    }

                    // Apply quadrature weights (from global - L1 cached)
                    const real_t wyz = wgll(iy) * wgll(iz);
                    const real_t wxz = wgll(ix) * wgll(iz);
                    const real_t wxy = wgll(ix) * wgll(iy);

                    const real_t fx = tmpx_x * wyz + tmpx_y * wxz + tmpx_z * wxy;
                    const real_t fy = tmpy_x * wyz + tmpy_y * wxz + tmpy_z * wxy;
                    const real_t fz = tmpz_x * wyz + tmpz_y * wxz + tmpz_z * wxy;

                    // Scatter to global force vector using atomic operations
                    AtomicAdd(d_f[gather_map_x(ix, iy, iz, ei)], -fx);
                    AtomicAdd(d_f[gather_map_y(ix, iy, iz, ei)], -fy);
                    AtomicAdd(d_f[gather_map_z(ix, iy, iz, ei)], -fz);
                }
            }
        }
        MFEM_SYNC_THREAD;

        // ===== Phase 5: Update memory variables and save strain =====
        // Recompute strain from s_disp (still in shared memory), then update M
        MFEM_FOREACH_THREAD(iz, z, NGLL)
        {
            MFEM_FOREACH_THREAD(iy, y, NGLL)
            {
                MFEM_FOREACH_THREAD(ix, x, NGLL)
                {
                    // Recompute reference gradients from shared memory
                    real_t tmpx_x = 0, tmpx_y = 0, tmpx_z = 0;
                    real_t tmpy_x = 0, tmpy_y = 0, tmpy_z = 0;
                    real_t tmpz_x = 0, tmpz_y = 0, tmpz_z = 0;

                    SEM_NOUNROLL
                    for (int k = 0; k < NGLL; k++)
                    {
                        const real_t hp_x = s_hprime[ix][k];
                        tmpx_x += s_disp_x[iz][iy][k] * hp_x;
                        tmpy_x += s_disp_y[iz][iy][k] * hp_x;
                        tmpz_x += s_disp_z[iz][iy][k] * hp_x;

                        const real_t hp_y = s_hprime[iy][k];
                        tmpx_y += s_disp_x[iz][k][ix] * hp_y;
                        tmpy_y += s_disp_y[iz][k][ix] * hp_y;
                        tmpz_y += s_disp_z[iz][k][ix] * hp_y;

                        const real_t hp_z = s_hprime[iz][k];
                        tmpx_z += s_disp_x[k][iy][ix] * hp_z;
                        tmpy_z += s_disp_y[k][iy][ix] * hp_z;
                        tmpz_z += s_disp_z[k][iy][ix] * hp_z;
                    }

                    // Load inverse Jacobian (will be L1/L2 cached from Phase 3)
                    const real_t j0 = invJPacked(0, ix, iy, iz, ei);
                    const real_t j1 = invJPacked(1, ix, iy, iz, ei);
                    const real_t j2 = invJPacked(2, ix, iy, iz, ei);
                    const real_t j3 = invJPacked(3, ix, iy, iz, ei);
                    const real_t j4 = invJPacked(4, ix, iy, iz, ei);
                    const real_t j5 = invJPacked(5, ix, iy, iz, ei);
                    const real_t j6 = invJPacked(6, ix, iy, iz, ei);
                    const real_t j7 = invJPacked(7, ix, iy, iz, ei);
                    const real_t j8 = invJPacked(8, ix, iy, iz, ei);

                    // Recompute strain
                    const real_t ekk = (tmpx_x * j0 + tmpx_y * j3 + tmpx_z * j6 +
                                        tmpy_x * j1 + tmpy_y * j4 + tmpy_z * j7 +
                                        tmpz_x * j2 + tmpz_y * j5 + tmpz_z * j8);
                    const real_t d11 = (tmpx_x * j0 + tmpx_y * j3 + tmpx_z * j6) - ekk / 3.0;
                    const real_t d22 = (tmpy_x * j1 + tmpy_y * j4 + tmpy_z * j7) - ekk / 3.0;
                    const real_t d12 = 0.5 * (tmpx_x * j1 + tmpx_y * j4 + tmpx_z * j7 +
                                              tmpy_x * j0 + tmpy_y * j3 + tmpy_z * j6);
                    const real_t d13 = 0.5 * (tmpx_x * j2 + tmpx_y * j5 + tmpx_z * j8 +
                                              tmpz_x * j0 + tmpz_y * j3 + tmpz_z * j6);
                    const real_t d23 = 0.5 * (tmpy_x * j2 + tmpy_y * j5 + tmpy_z * j8 +
                                              tmpz_x * j1 + tmpz_y * j4 + tmpz_z * j7);

                    // Load old strain from packed array
                    const real_t ekk_old = strain_old(0, ix, iy, iz, ei);
                    const real_t d11_old = strain_old(1, ix, iy, iz, ei);
                    const real_t d22_old = strain_old(2, ix, iy, iz, ei);
                    const real_t d12_old = strain_old(3, ix, iy, iz, ei);
                    const real_t d13_old = strain_old(4, ix, iy, iz, ei);
                    const real_t d23_old = strain_old(5, ix, iy, iz, ei);

                    // Update memory variables (Crank-Nicolson)
                    SEM_NOUNROLL
                    for (int uu = 0; uu < n_units; uu++)
                    {
                        // Load packed coefficients
                        const real_t alpha_kappa = coeffs(0, ix, iy, iz, ei, uu);
                        const real_t alpha_mu = coeffs(1, ix, iy, iz, ei, uu);
                        const real_t sc_kappa = coeffs(2, ix, iy, iz, ei, uu);
                        const real_t sc_mu = coeffs(3, ix, iy, iz, ei, uu);

                        // Crank-Nicolson: M_new = alpha * M_old + strain_coeff * (strain_old + strain_new)
                        M(0, ix, iy, iz, ei, uu) = alpha_kappa * M(0, ix, iy, iz, ei, uu) + sc_kappa * (ekk_old + ekk);
                        M(1, ix, iy, iz, ei, uu) = alpha_mu * M(1, ix, iy, iz, ei, uu) + sc_mu * (d11_old + d11);
                        M(2, ix, iy, iz, ei, uu) = alpha_mu * M(2, ix, iy, iz, ei, uu) + sc_mu * (d22_old + d22);
                        M(3, ix, iy, iz, ei, uu) = alpha_mu * M(3, ix, iy, iz, ei, uu) + sc_mu * (d12_old + d12);
                        M(4, ix, iy, iz, ei, uu) = alpha_mu * M(4, ix, iy, iz, ei, uu) + sc_mu * (d13_old + d13);
                        M(5, ix, iy, iz, ei, uu) = alpha_mu * M(5, ix, iy, iz, ei, uu) + sc_mu * (d23_old + d23);
                    }

                    // Save current strain for next time step
                    strain_old(0, ix, iy, iz, ei) = ekk;
                    strain_old(1, ix, iy, iz, ei) = d11;
                    strain_old(2, ix, iy, iz, ei) = d22;
                    strain_old(3, ix, iy, iz, ei) = d12;
                    strain_old(4, ix, iy, iz, ei) = d13;
                    strain_old(5, ix, iy, iz, ei) = d23;
                }
            }
        }
    });  // End of forall_3D
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

#define INSTANTIATE_VISCO_3D(NGLL) \
    template void SEMVisco_IsotropicElasticIntegrator3D::AddMultPA_Visco_Opt<NGLL>(const Vector&, Vector&) const;

SEM_INSTANTIATE_NGLL_3D(INSTANTIATE_VISCO_3D)

#undef INSTANTIATE_VISCO_3D

}  // namespace SEM
