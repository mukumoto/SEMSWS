/**
 * @file IsotropicElasticKernels3D.cpp
 * @brief 3D isotropic elastic stiffness kernel implementations
 *
 * GPU-optimized implementation using forall_3D + shared memory.
 * Each element gets NGLL³ threads for parallel computation.
 *
 * Memory layout: (ix, iy, iz, element) - Element is SLOWEST (matches libCEED/MFEM).
 */

#include "integ/SEMIsotropicElasticIntegrator.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "common/GpuMacros.hpp"
#include "general/forall.hpp"

namespace SEM {

// =============================================================================
// 3D Isotropic Elastic Kernel (GPU-Optimized: forall_3D + Shared Memory)
// =============================================================================

template<int NGLL>
void SEMIsotropicElasticIntegrator3D::AddMultPA_Opt(const Vector &u, Vector &f) const
{
    const int ne = ne_;

    const real_t* d_u = u.Read();
    real_t* d_f = f.ReadWrite();

    // DOF ordering views via factory: gather_map_x(ix, iy, iz, e)
    auto gather_map_x = dofs_.ViewGatherMapX();
    auto gather_map_y = dofs_.ViewGatherMapY();
    auto gather_map_z = dofs_.ViewGatherMapZ();

    // Material params view: params(comp, ix, iy, iz, e) where comp: 0=lambda, 1=mu
    auto params = ViewMaterialParams();

    // Packed inverse Jacobian + detJ view: invJPacked(comp, ix, iy, iz, e)
    // Components: 0-8 = invJ (row-major), 9 = detJ
    //   comp 0 = invJ[0][0] = (J11*J22 - J12*J21) / det
    //   comp 1 = invJ[0][1] = (J02*J21 - J01*J22) / det
    //   comp 2 = invJ[0][2] = (J01*J12 - J02*J11) / det
    //   comp 3 = invJ[1][0] = (J12*J20 - J10*J22) / det
    //   comp 4 = invJ[1][1] = (J00*J22 - J02*J20) / det
    //   comp 5 = invJ[1][2] = (J02*J10 - J00*J12) / det
    //   comp 6 = invJ[2][0] = (J10*J21 - J11*J20) / det
    //   comp 7 = invJ[2][1] = (J01*J20 - J00*J21) / det
    //   comp 8 = invJ[2][2] = (J00*J11 - J01*J10) / det
    //   comp 9 = detJ
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

    // GPU-optimized element loop: NGLL³ threads per element
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

        // Weak form coefficients (9 components for 3D)
        MFEM_SHARED real_t s_coeff[9][NGLL][NGLL][NGLL];

        // ===== Phase 1: Load basis functions to shared memory =====
        // Only threads in the z=0 plane load the 2D basis arrays
        // Original access: dxshape(k, ix) = data[k + ix * ngll]
        // Store as s_hprime[ix][k] so we access s_hprime[ix][k]
        if (MFEM_THREAD_ID(z) == 0)
        {
            MFEM_FOREACH_THREAD(iy, y, NGLL)
            {
                MFEM_FOREACH_THREAD(ix, x, NGLL)
                {
                    // s_hprime[ix][k]: for gradient computation (dshape(k, ix))
                    // s_hprime_w[k][ix]: for divergence computation (dshape_w(ix, k))
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
        MFEM_SYNC_THREAD;  // Ensure all displacements and basis loaded

        // ===== Phase 3: Gradient → Stress → Weak form coefficients (unified) =====
        // Computes: reference gradients → physical gradients → stress → weak form
        // All in one phase to minimize syncthreads and maximize register reuse
        MFEM_FOREACH_THREAD(iz, z, NGLL)
        {
            MFEM_FOREACH_THREAD(iy, y, NGLL)
            {
                MFEM_FOREACH_THREAD(ix, x, NGLL)
                {
                    // Step 1: Compute reference gradients from displacement
                    real_t dux_xi = 0, dux_eta = 0, dux_zeta = 0;
                    real_t duy_xi = 0, duy_eta = 0, duy_zeta = 0;
                    real_t duz_xi = 0, duz_eta = 0, duz_zeta = 0;

                    SEM_NOUNROLL
                    for (int k = 0; k < NGLL; k++)
                    {
                        const real_t hp_x = s_hprime[ix][k];
                        dux_xi   += s_disp_x[iz][iy][k] * hp_x;
                        duy_xi   += s_disp_y[iz][iy][k] * hp_x;
                        duz_xi   += s_disp_z[iz][iy][k] * hp_x;

                        const real_t hp_y = s_hprime[iy][k];
                        dux_eta  += s_disp_x[iz][k][ix] * hp_y;
                        duy_eta  += s_disp_y[iz][k][ix] * hp_y;
                        duz_eta  += s_disp_z[iz][k][ix] * hp_y;



                        const real_t hp_z = s_hprime[iz][k];                        
                        dux_zeta += s_disp_x[k][iy][ix] * hp_z;
                        duy_zeta += s_disp_y[k][iy][ix] * hp_z;
                        duz_zeta += s_disp_z[k][iy][ix] * hp_z;
                    }

                    // Step 2: Load inverse Jacobian and detJ from packed layout
                    // Packed layout: invJPacked(comp, ix, iy, iz, e)
                    // All 10 values are contiguous in memory for each (ix, iy, iz, e)
                    const real_t j0 = invJPacked(0, ix, iy, iz, ei);
                    const real_t j1 = invJPacked(1, ix, iy, iz, ei);
                    const real_t j2 = invJPacked(2, ix, iy, iz, ei);
                    const real_t j3 = invJPacked(3, ix, iy, iz, ei);
                    const real_t j4 = invJPacked(4, ix, iy, iz, ei);
                    const real_t j5 = invJPacked(5, ix, iy, iz, ei);
                    const real_t j6 = invJPacked(6, ix, iy, iz, ei);
                    const real_t j7 = invJPacked(7, ix, iy, iz, ei);
                    const real_t j8 = invJPacked(8, ix, iy, iz, ei);


                    // Step 4: Compute stress (isotropic elastic constitutive law)
                    const real_t dux_x = dux_xi * j0 + dux_eta * j3 + dux_zeta * j6;
                    const real_t duy_y = duy_xi * j1 + duy_eta * j4 + duy_zeta * j7;
                    const real_t duz_z = duz_xi * j2 + duz_eta * j5 + duz_zeta * j8;

                    const real_t lambda = params(0, ix, iy, iz, ei);
                    const real_t mu = params(1, ix, iy, iz, ei);
                    const real_t two_mu = mu + mu;

                    const real_t sigmaxx = lambda*(dux_x + duy_y + duz_z) + two_mu * dux_x;
                    const real_t sigmayy = lambda*(dux_x + duy_y + duz_z) + two_mu * duy_y;
                    const real_t sigmazz = lambda*(dux_x + duy_y + duz_z) + two_mu * duz_z;

                    const real_t dux_y = dux_xi * j1 + dux_eta * j4 + dux_zeta * j7;
                    const real_t dux_z = dux_xi * j2 + dux_eta * j5 + dux_zeta * j8;
                    const real_t duy_x = duy_xi * j0 + duy_eta * j3 + duy_zeta * j6;
                    const real_t duy_z = duy_xi * j2 + duy_eta * j5 + duy_zeta * j8;
                    const real_t duz_x = duz_xi * j0 + duz_eta * j3 + duz_zeta * j6;
                    const real_t duz_y = duz_xi * j1 + duz_eta * j4 + duz_zeta * j7;

                    const real_t sigmaxy = mu * (dux_y + duy_x);
                    const real_t sigmaxz = mu * (dux_z + duz_x);
                    const real_t sigmayz = mu * (duy_z + duz_y);


                    // Step 5: Compute weak form coefficients (sigma * invJ^T * detJ)
                    const real_t w = invJPacked(9, ix, iy, iz, ei);
                    // xi-direction (for x-derivative)
                    s_coeff[0][iz][iy][ix] = (sigmaxx * j0 + sigmaxy * j1 + sigmaxz * j2) * w;
                    s_coeff[5][iz][iy][ix] = (sigmaxy * j0 + sigmayy * j1 + sigmayz * j2) * w;
                    s_coeff[7][iz][iy][ix] = (sigmaxz * j0 + sigmayz * j1 + sigmazz * j2) * w;

                    // eta-direction (for y-derivative)
                    s_coeff[3][iz][iy][ix] = (sigmaxx * j3 + sigmaxy * j4 + sigmaxz * j5) * w;
                    s_coeff[1][iz][iy][ix] = (sigmaxy * j3 + sigmayy * j4 + sigmayz * j5) * w;
                    s_coeff[8][iz][iy][ix] = (sigmaxz * j3 + sigmayz * j4 + sigmazz * j5) * w;

                    // zeta-direction (for z-derivative)
                    s_coeff[4][iz][iy][ix] = (sigmaxx * j6 + sigmaxy * j7 + sigmaxz * j8) * w;
                    s_coeff[6][iz][iy][ix] = (sigmaxy * j6 + sigmayy * j7 + sigmayz * j8) * w;
                    s_coeff[2][iz][iy][ix] = (sigmaxz * j6 + sigmayz * j7 + sigmazz * j8) * w;
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
                    // Compute divergence via transpose of gradient operator
                    real_t tmpx_x = 0, tmpx_y = 0, tmpx_z = 0;
                    real_t tmpy_x = 0, tmpy_y = 0, tmpy_z = 0;
                    real_t tmpz_x = 0, tmpz_y = 0, tmpz_z = 0;

                    SEM_NOUNROLL
                    for (int k = 0; k < NGLL; k++)
                    {
                        // Original: nc[0][k][iy][iz] * dxshape_w(ix, k)
                        // s_coeff stored as [iz][iy][ix], so s_coeff[0][iz][iy][k] scans x
                        // s_hprime_w[k][ix] = dxshape_w(ix, k), so access s_hprime_w[k][ix]
                        const real_t hpw_x = s_hprime_w[k][ix];
                        tmpx_x += s_coeff[0][iz][iy][k] * hpw_x;
                        tmpy_x += s_coeff[5][iz][iy][k] * hpw_x;
                        tmpz_x += s_coeff[7][iz][iy][k] * hpw_x;

                        // Original: nc[3][ix][k][iz] * dyshape_w(iy, k)
                        const real_t hpw_y = s_hprime_w[k][iy];
                        tmpx_y += s_coeff[3][iz][k][ix] * hpw_y;
                        tmpy_y += s_coeff[1][iz][k][ix] * hpw_y;
                        tmpz_y += s_coeff[8][iz][k][ix] * hpw_y;

                        // Original: nc[4][ix][iy][k] * dzshape_w(iz, k)
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
                    // Required for GPU: multiple elements may share DOFs at element boundaries
                    // Negative sign: compute -K*u directly (eliminates separate negate kernel)
                    AtomicAdd(d_f[gather_map_x(ix, iy, iz, ei)], -fx);
                    AtomicAdd(d_f[gather_map_y(ix, iy, iz, ei)], -fy);
                    AtomicAdd(d_f[gather_map_z(ix, iy, iz, ei)], -fz);
                }
            }
        }
    });  // End of forall_3D
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

#define INSTANTIATE_ELASTIC_3D(NGLL) \
    template void SEMIsotropicElasticIntegrator3D::AddMultPA_Opt<NGLL>(const Vector&, Vector&) const;

SEM_INSTANTIATE_NGLL_3D(INSTANTIATE_ELASTIC_3D)

#undef INSTANTIATE_ELASTIC_3D

}  // namespace SEM
