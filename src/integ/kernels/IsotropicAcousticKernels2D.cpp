/**
 * @file IsotropicAcousticKernels2D.cpp
 * @brief 2D isotropic acoustic stiffness kernel implementations
 *
 * GPU-optimized implementation using forall_2D + shared memory.
 * Each element gets NGLL² threads for parallel computation.
 *
 * Memory layout: (ix, iy, element) - Element is SLOWEST (matches libCEED/MFEM).
 */

#include "integ/SEMIsotropicAcousticIntegrator.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "integ/kernels/AcousticFluxKernel2D.hpp"
#include "common/GpuMacros.hpp"
#include "general/forall.hpp"

namespace SEM {

// =============================================================================
// 2D Acoustic Kernel (GPU-Optimized: forall_2D + Shared Memory)
// =============================================================================

template<int NGLL>
void SEMIsotropicAcousticIntegrator2D::AddMultPA_Opt(const Vector &p, Vector &f) const
{
    const int ne = ne_;

    const real_t* d_p = p.Read();
    real_t* d_f = f.ReadWrite();

    // DOF ordering view via factory: gather_map(ix, iy, e)
    auto gather_map = dofs_.ViewGatherMap();

    // Material view: inv_rho(ix, iy, e)
    auto inv_rho = ViewInvRho();

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

    // GPU-optimized element loop: NGLL² threads per element
    mfem::forall_2D(ne, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei)
    {
        // ===== Shared memory for element data =====
        MFEM_SHARED real_t s_pres[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime_w[NGLL][NGLL];

        // Flux coefficients (2 components for 2D)
        MFEM_SHARED real_t s_flux[2][NGLL][NGLL];

        // ===== Phase 1: Load basis functions to shared memory =====
        // Only y=0 threads load (like z=0 in 3D Elastic)
        if (MFEM_THREAD_ID(y) == 0)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                for (int k = 0; k < NGLL; k++)
                {
                    s_hprime[ix][k] = dshape(k, ix);
                    s_hprime_w[ix][k] = dshape_w(k, ix);
                }
            }
        }

        // ===== Phase 2: Gather pressure from global to shared =====
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                s_pres[iy][ix] = d_p[gather_map(ix, iy, ei)];
            }
        }
        MFEM_SYNC_THREAD;

        // ===== Phase 3: Gradient → Flux → Weak form coefficients =====
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                // Compute reference gradients
                real_t dp_xi = 0, dp_eta = 0;

                SEM_NOUNROLL
                for (int k = 0; k < NGLL; k++)
                {
                    const real_t hp_x = s_hprime[ix][k];
                    dp_xi += s_pres[iy][k] * hp_x;

                    const real_t hp_y = s_hprime[iy][k];
                    dp_eta += s_pres[k][ix] * hp_y;
                }

                // Load inverse Jacobian and detJ from packed layout
                const real_t j0 = invJPacked(0, ix, iy, ei);
                const real_t j1 = invJPacked(1, ix, iy, ei);
                const real_t j2 = invJPacked(2, ix, iy, ei);
                const real_t j3 = invJPacked(3, ix, iy, ei);
                const real_t w = invJPacked(4, ix, iy, ei);

                // Compute weighted reference-frame flux via the shared
                // template (same math as before, now reusable by AD path).
                const real_t inv_rho_val = inv_rho(ix, iy, ei);
                real_t flux_xi, flux_eta;
                AcousticFlux2D<real_t>(inv_rho_val, dp_xi, dp_eta,
                                        j0, j1, j2, j3, w,
                                        flux_xi, flux_eta);
                s_flux[0][iy][ix] = flux_xi;
                s_flux[1][iy][ix] = flux_eta;
            }
        }
        MFEM_SYNC_THREAD;

        // ===== Phase 4: Compute force and scatter to global =====
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                real_t sum_x = 0, sum_y = 0;

                SEM_NOUNROLL
                for (int k = 0; k < NGLL; k++)
                {
                    const real_t hpw_x = s_hprime_w[k][ix];
                    sum_x += s_flux[0][iy][k] * hpw_x;

                    const real_t hpw_y = s_hprime_w[k][iy];
                    sum_y += s_flux[1][k][ix] * hpw_y;
                }

                const real_t wy = wgll(iy);
                const real_t wx = wgll(ix);

                const real_t force = sum_x * wy + sum_y * wx;
                // Negative sign: compute -K*u directly (eliminates separate negate kernel)
                AtomicAdd(d_f[gather_map(ix, iy, ei)], -force);
            }
        }
    });  // End of forall_2D
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

#define INSTANTIATE_ACOUSTIC_2D(NGLL) \
    template void SEMIsotropicAcousticIntegrator2D::AddMultPA_Opt<NGLL>(const Vector&, Vector&) const;

SEM_INSTANTIATE_NGLL_2D(INSTANTIATE_ACOUSTIC_2D)

#undef INSTANTIATE_ACOUSTIC_2D

}  // namespace SEM
