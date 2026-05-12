/**
 * @file IsotropicAcousticKernels3D.cpp
 * @brief 3D isotropic acoustic stiffness kernel implementations
 *
 * Uses MFEM-style DeviceTensor views for zero-overhead multi-dimensional access.
 * Memory layout: (ix, iy, iz, element) - Element is SLOWEST (matches libCEED/MFEM).
 */

#include "integ/SEMIsotropicAcousticIntegrator.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "common/GpuMacros.hpp"
#include <mfem.hpp>

namespace SEM {

// =============================================================================
// 3D Acoustic Kernel (View-Based)
// =============================================================================

template<int NGLL>
void SEMIsotropicAcousticIntegrator3D::AddMultPA_Opt(const Vector &p, Vector &f) const
{
    const int ne = ne_;

    // All captured by value into GPU registers
    auto d_p = p.Read();
    auto d_f = f.ReadWrite();

    // DOF ordering view via factory: gather_map(ix, iy, iz, e)
    auto gather_map = dofs_.ViewGatherMap();

    // Material view
    auto inv_rho = ViewInvRho();

    // Packed inverse Jacobian + detJ view: invJPacked(comp, ix, iy, iz, e)
    // Components: 0-8 = invJ (row-major), 9 = detJ
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
        // Layout [iz][iy][ix]: ix is innermost → contiguous for warp threads
        // (MFEM_FOREACH_THREAD(x) varies fastest → coalesced access)
        MFEM_SHARED real_t s_pot[NGLL][NGLL][NGLL];
        MFEM_SHARED real_t s_hprime[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime_w[NGLL][NGLL];

        // Flux coefficients (3 components for 3D)
        MFEM_SHARED real_t s_flux[3][NGLL][NGLL][NGLL];

        // ===== Phase 1: Load basis functions to shared memory =====
        // 2D arrays, only threads in the z=0 plane participate
        // s_hprime[ix][k] ≡ dshape(k, ix) = ℓ'_k(ξ_ix)   (row-major ↔ column-major flip)
        if (MFEM_THREAD_ID(z) == 0)
        {
            MFEM_FOREACH_THREAD(iy, y, NGLL)
            {
                MFEM_FOREACH_THREAD(ix, x, NGLL)
                {
                    s_hprime[ix][iy]   = dshape(iy, ix);
                    s_hprime_w[ix][iy] = dshape_w(iy, ix);
                }
            }
        }

        // ===== Phase 2: Gather potential from global to shared =====
        MFEM_FOREACH_THREAD(iz, z, NGLL)
        {
            MFEM_FOREACH_THREAD(iy, y, NGLL)
            {
                MFEM_FOREACH_THREAD(ix, x, NGLL)
                {
                    s_pot[iz][iy][ix] = d_p[gather_map(ix, iy, iz, ei)];
                }
            }
        }
        MFEM_SYNC_THREAD;

        // ===== Phase 3: Gradient → physical flux → weak-form coefficients =====
        MFEM_FOREACH_THREAD(iz, z, NGLL)
        {
            MFEM_FOREACH_THREAD(iy, y, NGLL)
            {
                MFEM_FOREACH_THREAD(ix, x, NGLL)
                {
                    // Reference gradients: ∂φ/∂(ξ,η,ζ)
                    real_t dchi_dxi = 0, dchi_deta = 0, dchi_dgamma = 0;
                    SEM_NOUNROLL
                    for (int k = 0; k < NGLL; k++)
                    {
                        const real_t hp_x = s_hprime[ix][k];
                        dchi_dxi    += s_pot[iz][iy][k] * hp_x;

                        const real_t hp_y = s_hprime[iy][k];
                        dchi_deta   += s_pot[iz][k][ix] * hp_y;

                        const real_t hp_z = s_hprime[iz][k];
                        dchi_dgamma += s_pot[k][iy][ix] * hp_z;
                    }

                    // Load inverse Jacobian and detJ from packed layout
                    const real_t j0 = invJPacked(0, ix, iy, iz, ei);
                    const real_t j1 = invJPacked(1, ix, iy, iz, ei);
                    const real_t j2 = invJPacked(2, ix, iy, iz, ei);
                    const real_t j3 = invJPacked(3, ix, iy, iz, ei);
                    const real_t j4 = invJPacked(4, ix, iy, iz, ei);
                    const real_t j5 = invJPacked(5, ix, iy, iz, ei);
                    const real_t j6 = invJPacked(6, ix, iy, iz, ei);
                    const real_t j7 = invJPacked(7, ix, iy, iz, ei);
                    const real_t j8 = invJPacked(8, ix, iy, iz, ei);
                    const real_t w  = invJPacked(9, ix, iy, iz, ei) * inv_rho(ix, iy, iz, ei);

                    // Physical gradient: grad(chi) = J^{-T} * grad_ref(chi)
                    // (row-major invJ: invJ[i][j] = j_{3i+j}; grad_x = Σ_r invJ[r][0] * dchi/dξ_r)
                    const real_t dchi_dx = dchi_dxi * j0 + dchi_deta * j3 + dchi_dgamma * j6;
                    const real_t dchi_dy = dchi_dxi * j1 + dchi_deta * j4 + dchi_dgamma * j7;
                    const real_t dchi_dz = dchi_dxi * j2 + dchi_deta * j5 + dchi_dgamma * j8;

                    // Weak form: J^{-1} * (1/rho) * grad(chi) * detJ
                    s_flux[0][iz][iy][ix] = (dchi_dx * j0 + dchi_dy * j1 + dchi_dz * j2) * w;
                    s_flux[1][iz][iy][ix] = (dchi_dx * j3 + dchi_dy * j4 + dchi_dz * j5) * w;
                    s_flux[2][iz][iy][ix] = (dchi_dx * j6 + dchi_dy * j7 + dchi_dz * j8) * w;
                }
            }
        }
        MFEM_SYNC_THREAD;

        // ===== Phase 4: Compute force and scatter =====
        MFEM_FOREACH_THREAD(iz, z, NGLL)
        {
            MFEM_FOREACH_THREAD(iy, y, NGLL)
            {
                MFEM_FOREACH_THREAD(ix, x, NGLL)
                {
                    real_t tmp_xi = 0, tmp_eta = 0, tmp_zeta = 0;
                    SEM_NOUNROLL
                    for (int k = 0; k < NGLL; k++)
                    {
                        tmp_xi   += s_flux[0][iz][iy][k] * s_hprime_w[k][ix];
                        tmp_eta  += s_flux[1][iz][k][ix] * s_hprime_w[k][iy];
                        tmp_zeta += s_flux[2][k][iy][ix] * s_hprime_w[k][iz];
                    }

                    const real_t wyz = wgll(iy) * wgll(iz);
                    const real_t wxz = wgll(ix) * wgll(iz);
                    const real_t wxy = wgll(ix) * wgll(iy);

                    const real_t force = tmp_xi * wyz + tmp_eta * wxz + tmp_zeta * wxy;

                    // Negative sign: compute -K*u directly (eliminates separate negate kernel)
                    AtomicAdd(d_f[gather_map(ix, iy, iz, ei)], -force);
                }
            }
        }
    });
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

#define INSTANTIATE_ACOUSTIC_3D(NGLL) \
    template void SEMIsotropicAcousticIntegrator3D::AddMultPA_Opt<NGLL>(const Vector&, Vector&) const;

SEM_INSTANTIATE_NGLL_3D(INSTANTIATE_ACOUSTIC_3D)

#undef INSTANTIATE_ACOUSTIC_3D

}  // namespace SEM
