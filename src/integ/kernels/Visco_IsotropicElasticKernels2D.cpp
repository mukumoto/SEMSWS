/**
 * @file Visco_IsotropicElasticKernels2D.cpp
 * @brief 2D viscoelastic (isotropic elastic with attenuation) stiffness kernel implementations
 *
 * GPU-optimized implementation using forall_2D + shared memory.
 * Each element gets NGLL² threads for parallel computation.
 *
 * Memory layout: (ix, iy, element) - Element is SLOWEST (matches libCEED/MFEM).
 * Memory variables (M1-M3) are updated per-GLL-point within the kernel.
 */

#include "integ/SEMVisco_IsotropicElasticIntegrator.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "common/GpuMacros.hpp"
#include "general/forall.hpp"

namespace SEM {

// =============================================================================
// 2D Viscoelastic Kernel (GPU-Optimized: forall_2D + Shared Memory)
// =============================================================================

template<int NGLL>
void SEMVisco_IsotropicElasticIntegrator2D::AddMultPA_Visco_Opt(const Vector &u, Vector &f) const
{
    const int ne = ne_;

    const real_t* d_u = u.Read();
    real_t* d_f = f.ReadWrite();

    // DOF ordering views via factory: gather_map_x(ix, iy, e)
    auto gather_map_x = dofs_.ViewGatherMapX();
    auto gather_map_y = dofs_.ViewGatherMapY();

    // Material params view: params(comp, ix, iy, e) where comp: 0=kappa, 1=mu
    auto params = ViewMaterialParams();

    // Packed inverse Jacobian + detJ view: invJPacked(comp, ix, iy, e)
    // Components: 0=j00, 1=j01, 2=j10, 3=j11, 4=detJ
    auto invJPacked = geom_.ViewInvJPacked();

    // Shape derivative views: dshape(k, ix) = ℓ'_k(ξ_ix)
    //   = derivative of the k-th 1D Lagrange basis evaluated at the ix-th GLL node.
    // Data stored row-major in geom_.dxshape[ngll * ngll]:
    //   dxshape[i * ngll + j] = ℓ'_j(ξ_i)
    //   (from SEMGeometry: quad_fe.CalcDShape at IntPoint(i) with lex[j]=(j,0);
    //    tensor-product basis makes dshape_mat(lex[j], 0) reduce to ℓ'_j(ξ_i))
    // MFEM Reshape is column-major: view(a, b) = data[a + b * ngll]
    //   → dshape(k, ix)   = data[ix * ngll + k] = ℓ'_k(ξ_ix)   (Phase 1 gradient)
    //   → dshape_w(ix, k) = data[k * ngll + ix] = ℓ'_ix(ξ_k)·w_k (Phase 2 scatter)
    // Same 1D basis for ξ/η directions (tensor product element).
    //
    // dshape_w is quadrature-weighted:
    //   dxshape_w[i * ngll + j] = ℓ'_j(ξ_i) · w_i   (w_i = GLL weight at node i)
    auto dshape = geom_.ViewDShape();
    auto dshape_w = geom_.ViewDShapeW();

    // Quadrature weight view (same for all directions in tensor product element)
    auto wgll = geom_.ViewWgll();

    // Get memory state from attenuation model
    ViscoelasticMemory2D* memory = const_cast<GeneralizedZener2D&>(attenuation_).GetMemory();
    MFEM_VERIFY(memory != nullptr, "Viscoelastic memory not initialized");

    const int n_units = memory->NumUnits();

    // Memory variable View [3, ngllx, nglly, ne, n_units]
    auto M = memory->ViewMWrite();

    // Old strain View [3, ngllx, nglly, ne]
    auto strain_old = memory->ViewStrainOldWrite();

    // Crank-Nicolson coefficient View [4, ngllx, nglly, ne, n_units]
    auto coeffs = memory->ViewCoeffs();

    // GPU-optimized element loop: NGLL² threads per element
    mfem::forall_2D(ne, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei)
    {
        // ===== Shared memory for element data =====
        MFEM_SHARED real_t s_disp_x[NGLL][NGLL];
        MFEM_SHARED real_t s_disp_y[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime_w[NGLL][NGLL];

        // Stress coefficients (4 components for 2D weak form)
        MFEM_SHARED real_t s_coeff[4][NGLL][NGLL];

        // ===== Phase 1: Load basis functions to shared memory (parallel, like 3D) =====
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                // Each thread loads one element: s_hprime[ix][iy] = dshape(iy, ix)
                s_hprime[ix][iy] = dshape(iy, ix);
                s_hprime_w[ix][iy] = dshape_w(iy, ix);
            }
        }

        // ===== Phase 2: Gather displacement from global to shared =====
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                s_disp_x[iy][ix] = d_u[gather_map_x(ix, iy, ei)];
                s_disp_y[iy][ix] = d_u[gather_map_y(ix, iy, ei)];
            }
        }
        MFEM_SYNC_THREAD;

        // ===== Phase 3: Compute stress with viscoelastic correction (READ ONLY M) =====
        // Only read M values to subtract from stress, do NOT update M here
        // M update is deferred to Phase 5 (after scatter) to reduce register pressure
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                // Compute reference gradients via sum factorization
                real_t tmpx_x = 0, tmpx_y = 0;
                real_t tmpy_x = 0, tmpy_y = 0;

                SEM_NOUNROLL
                for (int k = 0; k < NGLL; k++)
                {
                    const real_t hp_x = s_hprime[ix][k];
                    tmpx_x += s_disp_x[iy][k] * hp_x;
                    tmpy_x += s_disp_y[iy][k] * hp_x;

                    const real_t hp_y = s_hprime[iy][k];
                    tmpx_y += s_disp_x[k][ix] * hp_y;
                    tmpy_y += s_disp_y[k][ix] * hp_y;
                }

                // Load inverse Jacobian from packed layout
                const real_t j0 = invJPacked(0, ix, iy, ei);
                const real_t j1 = invJPacked(1, ix, iy, ei);
                const real_t j2 = invJPacked(2, ix, iy, ei);
                const real_t j3 = invJPacked(3, ix, iy, ei);
                const real_t w = invJPacked(4, ix, iy, ei);

                // Physical gradients
                const real_t dux_x = tmpx_x * j0 + tmpx_y * j3;
                const real_t duy_y = tmpy_x * j2 + tmpy_y * j1;
                const real_t duy_x = tmpy_x * j0 + tmpy_y * j3;
                const real_t dux_y = tmpx_x * j2 + tmpx_y * j1;

                // Load material parameters
                const real_t kappa = params(0, ix, iy, ei);
                const real_t mu_val = params(1, ix, iy, ei);

                // Strain components for viscoelasticity
                const real_t ekk = dux_x + duy_y;
                const real_t d11 = dux_x - ekk / 2.0;
                const real_t d12 = 0.5 * (dux_y + duy_x);

                // Elastic stress using UNRELAXED moduli
                real_t sigmaxx = kappa * ekk + 2.0 * mu_val * d11;
                real_t sigmayy = kappa * ekk - 2.0 * mu_val * d11;
                real_t sigmaxy = 2.0 * mu_val * d12;

                // Sum M contributions (READ ONLY - no update here)
                SEM_NOUNROLL
                for (int uu = 0; uu < n_units; uu++)
                {
                    const real_t m1 = M(0, ix, iy, ei, uu);
                    const real_t m2 = M(1, ix, iy, ei, uu);
                    const real_t m3 = M(2, ix, iy, ei, uu);

                    sigmaxx -= kappa * m1 + mu_val * m2;
                    sigmayy -= kappa * m1 - mu_val * m2;
                    sigmaxy -= mu_val * m3;
                }

                // Weak form back-transformation: sigma · J^{-T} * detJ
                s_coeff[0][iy][ix] = (sigmaxx * j0 + sigmaxy * j2) * w;
                s_coeff[1][iy][ix] = (sigmaxy * j0 + sigmayy * j2) * w;
                s_coeff[2][iy][ix] = (sigmaxx * j3 + sigmaxy * j1) * w;
                s_coeff[3][iy][ix] = (sigmaxy * j3 + sigmayy * j1) * w;
            }
        }
        MFEM_SYNC_THREAD;

        // ===== Phase 4: Compute force and scatter to global =====
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                real_t tmpx_x = 0, tmpx_y = 0;
                real_t tmpy_x = 0, tmpy_y = 0;

                SEM_NOUNROLL
                for (int k = 0; k < NGLL; k++)
                {
                    const real_t hpw_x = s_hprime_w[k][ix];
                    tmpx_x += s_coeff[0][iy][k] * hpw_x;
                    tmpy_x += s_coeff[1][iy][k] * hpw_x;

                    const real_t hpw_y = s_hprime_w[k][iy];
                    tmpx_y += s_coeff[2][k][ix] * hpw_y;
                    tmpy_y += s_coeff[3][k][ix] * hpw_y;
                }

                const real_t wy = wgll(iy);
                const real_t wx = wgll(ix);

                const real_t fx = tmpx_x * wy + tmpx_y * wx;
                const real_t fy = tmpy_x * wy + tmpy_y * wx;

                AtomicAdd(d_f[gather_map_x(ix, iy, ei)], -fx);
                AtomicAdd(d_f[gather_map_y(ix, iy, ei)], -fy);
            }
        }
        MFEM_SYNC_THREAD;

        // ===== Phase 5: Update memory variables and save strain =====
        // Recompute strain from s_disp (still in shared memory), then update M
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                // Recompute reference gradients from shared memory
                real_t tmpx_x = 0, tmpx_y = 0;
                real_t tmpy_x = 0, tmpy_y = 0;

                SEM_NOUNROLL
                for (int k = 0; k < NGLL; k++)
                {
                    const real_t hp_x = s_hprime[ix][k];
                    tmpx_x += s_disp_x[iy][k] * hp_x;
                    tmpy_x += s_disp_y[iy][k] * hp_x;

                    const real_t hp_y = s_hprime[iy][k];
                    tmpx_y += s_disp_x[k][ix] * hp_y;
                    tmpy_y += s_disp_y[k][ix] * hp_y;
                }

                // Load inverse Jacobian (will be L1/L2 cached from Phase 3)
                const real_t j0 = invJPacked(0, ix, iy, ei);
                const real_t j1 = invJPacked(1, ix, iy, ei);
                const real_t j2 = invJPacked(2, ix, iy, ei);
                const real_t j3 = invJPacked(3, ix, iy, ei);

                // Recompute physical gradients
                const real_t dux_x = tmpx_x * j0 + tmpx_y * j3;
                const real_t duy_y = tmpy_x * j2 + tmpy_y * j1;
                const real_t duy_x = tmpy_x * j0 + tmpy_y * j3;
                const real_t dux_y = tmpx_x * j2 + tmpx_y * j1;

                // Recompute strain
                const real_t ekk = dux_x + duy_y;
                const real_t d11 = dux_x - ekk / 2.0;
                const real_t d12 = 0.5 * (dux_y + duy_x);

                // Load old strain from packed array
                const real_t ekk_old = strain_old(0, ix, iy, ei);
                const real_t d11_old = strain_old(1, ix, iy, ei);
                const real_t d12_old = strain_old(2, ix, iy, ei);

                // Update memory variables (Crank-Nicolson)
                SEM_NOUNROLL
                for (int uu = 0; uu < n_units; uu++)
                {
                    const real_t alpha_kappa = coeffs(0, ix, iy, ei, uu);
                    const real_t alpha_mu = coeffs(1, ix, iy, ei, uu);
                    const real_t sc_kappa = coeffs(2, ix, iy, ei, uu);
                    const real_t sc_mu = coeffs(3, ix, iy, ei, uu);

                    M(0, ix, iy, ei, uu) = alpha_kappa * M(0, ix, iy, ei, uu) + sc_kappa * (ekk_old + ekk);
                    M(1, ix, iy, ei, uu) = alpha_mu * M(1, ix, iy, ei, uu) + sc_mu * (d11_old + d11);
                    M(2, ix, iy, ei, uu) = alpha_mu * M(2, ix, iy, ei, uu) + sc_mu * (d12_old + d12);
                }

                // Save current strain for next time step
                strain_old(0, ix, iy, ei) = ekk;
                strain_old(1, ix, iy, ei) = d11;
                strain_old(2, ix, iy, ei) = d12;
            }
        }
    });  // End of forall_2D
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

#define INSTANTIATE_VISCO_2D(NGLL) \
    template void SEMVisco_IsotropicElasticIntegrator2D::AddMultPA_Visco_Opt<NGLL>(const Vector&, Vector&) const;

SEM_INSTANTIATE_NGLL_2D(INSTANTIATE_VISCO_2D)

#undef INSTANTIATE_VISCO_2D

}  // namespace SEM
