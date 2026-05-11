/**
 * @file IsotropicElasticKernels2D.cpp
 * @brief 2D isotropic elastic stiffness kernel implementations
 *
 * GPU-optimized implementation using forall_2D + shared memory.
 * Each element gets NGLL² threads for parallel computation.
 *
 * Memory layout: (ix, iy, element) - Element is SLOWEST (matches libCEED/MFEM).
 */

#include "integ/SEMIsotropicElasticIntegrator.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "common/GpuMacros.hpp"
#include "general/forall.hpp"

namespace SEM {

// =============================================================================
// 2D Isotropic Elastic Kernel (GPU-Optimized: forall_2D + Shared Memory)
// =============================================================================

template<int NGLL>
void SEMIsotropicElasticIntegrator2D::AddMultPA_Opt(const Vector &u, Vector &f) const
{
    const int ne = ne_;

    const real_t* d_u = u.Read();
    real_t* d_f = f.ReadWrite();

    // DOF ordering views via factory: gather_map_x(ix, iy, e)
    auto gather_map_x = dofs_.ViewGatherMapX();
    auto gather_map_y = dofs_.ViewGatherMapY();

    // Material params view: params(comp, ix, iy, e) where comp: 0=lambda, 1=mu
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

    // GPU-optimized element loop: NGLL² threads per element
    mfem::forall_2D(ne, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei)
    {
        // ===== Shared memory for element data =====
        // On GPU: __shared__ memory, fast access within thread-block
        // On CPU: Just local arrays (stack or thread-private)
        MFEM_SHARED real_t s_disp_x[NGLL][NGLL];
        MFEM_SHARED real_t s_disp_y[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime_w[NGLL][NGLL];

        // Weak form coefficients (4 components for 2D)
        MFEM_SHARED real_t s_coeff[4][NGLL][NGLL];

        // ===== Phase 1: Load basis functions to shared memory (parallel, like 3D) =====
        // Each thread loads one element: s_hprime[ix][iy] = dshape(iy, ix)
        // s_hprime[ix][k]: for gradient computation (dshape(k, ix))
        // s_hprime_w[k][ix]: for divergence computation (dshape_w(ix, k))
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
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
        MFEM_SYNC_THREAD;  // Ensure all displacements and basis loaded

        // ===== Phase 3: Gradient → Stress → Weak form coefficients (unified) =====
        // Computes: reference gradients → physical gradients → stress → weak form
        // All in one phase to minimize syncthreads and maximize register reuse
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                // Step 1: Compute reference gradients from displacement
                real_t dux_xi = 0, dux_eta = 0;
                real_t duy_xi = 0, duy_eta = 0;

                SEM_NOUNROLL
                for (int k = 0; k < NGLL; k++)
                {
                    const real_t hp_x = s_hprime[ix][k];
                    dux_xi += s_disp_x[iy][k] * hp_x;
                    duy_xi += s_disp_y[iy][k] * hp_x;

                    const real_t hp_y = s_hprime[iy][k];
                    dux_eta += s_disp_x[k][ix] * hp_y;
                    duy_eta += s_disp_y[k][ix] * hp_y;
                }

                // Step 2: Load inverse Jacobian and detJ from packed layout
                // Packed layout: invJPacked(comp, ix, iy, e)
                const real_t j0 = invJPacked(0, ix, iy, ei);
                const real_t j1 = invJPacked(1, ix, iy, ei);
                const real_t j2 = invJPacked(2, ix, iy, ei);
                const real_t j3 = invJPacked(3, ix, iy, ei);
                const real_t w = invJPacked(4, ix, iy, ei);

                // Step 3: Compute physical gradients
                const real_t dux_dx = dux_xi * j0 + dux_eta * j3;
                const real_t duy_dy = duy_xi * j2 + duy_eta * j1;
                const real_t duy_dx = duy_xi * j0 + duy_eta * j3;
                const real_t dux_dy = dux_xi * j2 + dux_eta * j1;

                // Step 4: Load material parameters and compute stress
                const real_t lambda = params(0, ix, iy, ei);
                const real_t mu = params(1, ix, iy, ei);
                const real_t two_mu = mu + mu;

                const real_t trace = dux_dx + duy_dy;
                const real_t sigmaxx = lambda * trace + two_mu * dux_dx;
                const real_t sigmayy = lambda * trace + two_mu * duy_dy;
                const real_t sigmaxy = mu * (dux_dy + duy_dx);

                // Step 5: Compute weak form coefficients (sigma * invJ^T * detJ)
                // xi-direction (for x-derivative)
                s_coeff[0][iy][ix] = (sigmaxx * j0 + sigmaxy * j2) * w;
                s_coeff[1][iy][ix] = (sigmaxy * j0 + sigmayy * j2) * w;

                // eta-direction (for y-derivative)
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
                // Compute divergence via transpose of gradient operator
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

                // Apply quadrature weights (from global - L1 cached)
                const real_t wy = wgll(iy);
                const real_t wx = wgll(ix);

                const real_t fx = tmpx_x * wy + tmpx_y * wx;
                const real_t fy = tmpy_x * wy + tmpy_y * wx;

                // Scatter to global force vector using atomic operations
                // Required for GPU: multiple elements may share DOFs at element boundaries
                // Negative sign: compute -K*u directly (eliminates separate negate kernel)
                AtomicAdd(d_f[gather_map_x(ix, iy, ei)], -fx);
                AtomicAdd(d_f[gather_map_y(ix, iy, ei)], -fy);
            }
        }
    });  // End of forall_2D
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

#define INSTANTIATE_ELASTIC_2D(NGLL) \
    template void SEMIsotropicElasticIntegrator2D::AddMultPA_Opt<NGLL>(const Vector&, Vector&) const;

SEM_INSTANTIATE_NGLL_2D(INSTANTIATE_ELASTIC_2D)

#undef INSTANTIATE_ELASTIC_2D

}  // namespace SEM
