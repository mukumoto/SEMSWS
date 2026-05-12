/**
 * @file Visco_IsotropicAcousticKernels2D.cpp
 * @brief 2D viscoacoustic stiffness kernel implementations (Crank-Nicolson scheme)
 *
 * GPU-optimized implementation using forall_2D + shared memory.
 * Each element gets NGLL² threads for parallel computation.
 * Memory variables are updated using Crank-Nicolson scheme matching 2D/3D viscoelastic.
 *
 * Crank-Nicolson scheme:
 *   e1_new = alpha × e1_old + beta × strain_old + gamma × strain_new
 *   where strain = stiffness operator output = -∇·((1/ρ)∇φ)
 */

#include "integ/SEMVisco_IsotropicAcousticIntegrator.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "common/GpuMacros.hpp"
#include <mfem.hpp>

namespace SEM {

// =============================================================================
// 2D Viscoacoustic Kernel (GPU-Optimized: forall_2D + Shared Memory)
// =============================================================================
template<int NGLL>
void SEMVisco_IsotropicAcousticIntegrator2D::AddMultPA_Visco_Opt(const Vector &phi, Vector &f) const
{
    const int ne = ne_;

    const real_t* d_phi = phi.Read();
    real_t* d_f = f.ReadWrite();

    // DOF ordering view via factory: gather_map(ix, iy, e)
    auto gather_map = dofs_.ViewGatherMap();

    // Material params view: params(comp, ix, iy, e) where comp: 0=inv_rho, 1=kappa
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
    ViscoacousticMemory2D* memory = const_cast<GeneralizedZenerAcoustic2D&>(attenuation_).GetMemory();
    if (memory == nullptr) {
        std::cerr << "FATAL: Viscoacoustic memory null in AddMultPA_Visco_Opt" << std::endl;
        std::cerr.flush();
    }
    MFEM_VERIFY(memory != nullptr, "Viscoacoustic memory not initialized (Opt)");

    const int n_units = memory->NumUnits();

    // Memory variable views
    // Layout: [ngllx, nglly, ne, n_units]
    auto e1 = memory->ViewE1();
    auto dot_e1_old = memory->ViewDotE1Old();
    // Crank-Nicolson coefficients: [2, ngllx, nglly, ne, n_units]
    // Components: 0=alpha, 1=strain_coeff
    auto coeffs = memory->ViewCoeffs();

    // GPU-optimized element loop: NGLL² threads per element
    mfem::forall_2D(ne, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei)
    {
        // ===== Shared memory for element data =====
        MFEM_SHARED real_t s_pot[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime_w[NGLL][NGLL];

        // Flux coefficients (2 components for 2D)
        MFEM_SHARED real_t s_flux[2][NGLL][NGLL];

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

        // ===== Phase 2: Gather potential from global to shared =====
        MFEM_FOREACH_THREAD(iy, y, NGLL)
        {
            MFEM_FOREACH_THREAD(ix, x, NGLL)
            {
                s_pot[iy][ix] = d_phi[gather_map(ix, iy, ei)];
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
                    dp_xi += s_pot[iy][k] * hp_x;

                    const real_t hp_y = s_hprime[iy][k];
                    dp_eta += s_pot[k][ix] * hp_y;
                }

                // Load inverse Jacobian and detJ from packed layout
                const real_t j0 = invJPacked(0, ix, iy, ei);
                const real_t j1 = invJPacked(1, ix, iy, ei);
                const real_t j2 = invJPacked(2, ix, iy, ei);
                const real_t j3 = invJPacked(3, ix, iy, ei);
                const real_t w = invJPacked(4, ix, iy, ei);

                // Physical gradients
                const real_t dp_dx = dp_xi * j0 + dp_eta * j3;
                const real_t dp_dy = dp_xi * j2 + dp_eta * j1;

                // Compute flux: q = (1/rho) * grad(φ)
                const real_t inv_rho_val = params(0, ix, iy, ei);
                const real_t qx = inv_rho_val * dp_dx;
                const real_t qy = inv_rho_val * dp_dy;

                // Transform back to reference
                s_flux[0][iy][ix] = (qx * j0 + qy * j2) * w;
                s_flux[1][iy][ix] = (qx * j3 + qy * j1) * w;
            }
        }
        MFEM_SYNC_THREAD;

        // ===== Phase 4: Compute stiffness output (strain), force with OLD memory, and scatter =====
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

                // This is the stiffness operator output (negative divergence of flux)
                const real_t strain_val = sum_x * wy + sum_y * wx;

                // Sum OLD memory variables (read before update, matches viscoelastic)
                real_t mem_sum = 0.0;
                SEM_NOUNROLL
                for (int uu = 0; uu < n_units; uu++) {
                    mem_sum += e1(ix, iy, ei, uu);
                }

                // Final force: -stiffness - memory (using OLD memory)
                AtomicAdd(d_f[gather_map(ix, iy, ei)], -strain_val - mem_sum);

                // ===== Update memory variables (Crank-Nicolson) =====
                // NOTE: Negate strain to compensate for sign difference in coefficients:
                //   Note: B_newmark = phi_nu1 * coef < 0 (phi_nu1 < 0)
                //   SEMSWS: beta = old_strain_coeff * norm_factor > 0 (norm_factor > 0)
                const real_t strain_new = -strain_val;
                const real_t strain_old_val = dot_e1_old(ix, iy, ei);

                SEM_NOUNROLL
                for (int uu = 0; uu < n_units; uu++) {
                    const real_t alpha = coeffs(0, ix, iy, ei, uu);
                    const real_t sc = coeffs(1, ix, iy, ei, uu);

                    // Crank-Nicolson: e1_new = alpha × e1_old + strain_coeff × (strain_old + strain_new)
                    e1(ix, iy, ei, uu) = alpha * e1(ix, iy, ei, uu)
                                       + sc * (strain_old_val + strain_new);
                }

                // Save strain for next timestep
                dot_e1_old(ix, iy, ei) = strain_new;
            }
        }
    });  // End of forall_2D
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

#define INSTANTIATE_VISCO_ACOUSTIC_2D(NGLL) \
    template void SEMVisco_IsotropicAcousticIntegrator2D::AddMultPA_Visco_Opt<NGLL>(const Vector&, Vector&) const;

SEM_INSTANTIATE_NGLL_2D(INSTANTIATE_VISCO_ACOUSTIC_2D)

#undef INSTANTIATE_VISCO_ACOUSTIC_2D

}  // namespace SEM
