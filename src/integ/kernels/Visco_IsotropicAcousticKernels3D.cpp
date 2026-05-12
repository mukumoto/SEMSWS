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
// 3D Viscoacoustic Kernel (GPU-Optimized: forall_3D + Shared Memory)
// =============================================================================
template<int NGLL>
void SEMVisco_IsotropicAcousticIntegrator3D::AddMultPA_Visco_Opt(const Vector &phi, Vector &f) const
{
    const int ne = ne_;

    const real_t* d_phi = phi.Read();
    real_t* d_f = f.ReadWrite();

    // DOF ordering view via factory: gather_map(ix, iy, iz, e)
    auto gather_map = dofs_.ViewGatherMap();

    // Material params view: params(comp, ix, iy, iz, e) where comp: 0=inv_rho, 1=kappa
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
    // Attenuation model is constant as this kernel is const.
    // Remove this const by const_cast
    ViscoacousticMemory3D* memory = const_cast<GeneralizedZenerAcoustic3D&>(attenuation_).GetMemory();
    if (memory == nullptr) {
        std::cerr << "FATAL: Viscoacoustic memory null in AddMultPA_Visco_Opt" << std::endl;
        std::cerr.flush();
    }
    MFEM_VERIFY(memory != nullptr, "Viscoacoustic memory not initialized (Opt)");

    const int n_units = memory->NumUnits();

    // Memory variable views
    // Layout: [ngllx, nglly, ngllz, ne, n_units]
    auto e1 = memory->ViewE1();
    auto dot_e1_old = memory->ViewDotE1Old();
    // Crank-Nicolson coefficients: [2, ngllx, nglly, ngllz, ne, n_units]
    // Components: 0=alpha, 1=strain_coeff
    auto coeffs = memory->ViewCoeffs();

    // GPU-optimized element loop: NGLL² threads per element
    mfem::forall_3D(ne, NGLL, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei)
    {
        // ===== Shared memory for element data =====
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
                    s_pot[iz][iy][ix] = d_phi[gather_map(ix, iy, iz, ei)];
                }  
            }
        }

        MFEM_SYNC_THREAD;  // Ensure all threads have loaded potential and basis functions before proceeding


        // ===== Phase 3: Gradient → Flux → Weak form coefficients =====
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
                    const real_t w  = invJPacked(9, ix, iy, iz, ei) * params(0, ix, iy, iz, ei);

                    // Physical gradients: grad(chi) = J^{-T} * grad_ref(chi)
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
        MFEM_SYNC_THREAD;  // Ensure all threads have computed flux before proceeding

        //Phase 4: Compute stiffness output (strain), force with OLD memory, and scatter
        MFEM_FOREACH_THREAD(iz, z, NGLL)
        {
            MFEM_FOREACH_THREAD(iy, y, NGLL)
            {
                MFEM_FOREACH_THREAD(ix, x, NGLL)
                {
                    real_t sum_x = 0, sum_y = 0, sum_z = 0;

                    SEM_NOUNROLL
                    for (int k = 0; k < NGLL; k++)
                    {
                        const real_t hpw_x = s_hprime_w[k][ix];
                        sum_x += s_flux[0][iz][iy][k] * hpw_x;

                        const real_t hpw_y = s_hprime_w[k][iy];
                        sum_y += s_flux[1][iz][k][ix] * hpw_y;

                        const real_t hpw_z = s_hprime_w[k][iz];
                        sum_z += s_flux[2][k][iy][ix] * hpw_z;
                    }

                    const real_t wz = wgll(iz);
                    const real_t wy = wgll(iy);
                    const real_t wx = wgll(ix);

                    // This is the stiffness operator output (negative divergence of flux)
                    const real_t strain_val = sum_x * wy * wz + sum_y * wx * wz + sum_z * wx * wy;

                    // Sum OLD memory variables (read before update, matches viscoelastic)
                    real_t mem_sum = 0.0;
                    SEM_NOUNROLL
                    for (int uu = 0; uu < n_units; uu++) {
                        mem_sum += e1(ix, iy, iz, ei, uu);
                    }

                    // Final force: -stiffness - memory (using OLD memory)
                    AtomicAdd(d_f[gather_map(ix, iy, iz, ei)], -strain_val - mem_sum);

                    // ===== Update memory variables (Crank-Nicolson) =====
                    // NOTE: Negate strain to compensate for sign difference in coefficients:
                    //   Note: B_newmark = phi_nu1 * coef < 0 (phi_nu1 < 0)
                    //   SEMSWS: beta = old_strain_coeff * norm_factor > 0 (norm_factor > 0)
                    const real_t strain_new = -strain_val;
                    const real_t strain_old_val = dot_e1_old(ix, iy, iz, ei);

                    SEM_NOUNROLL
                    for (int uu = 0; uu < n_units; uu++) {
                        const real_t alpha = coeffs(0, ix, iy, iz, ei, uu);
                        const real_t sc = coeffs(1, ix, iy, iz, ei, uu);
                        // Crank-Nicolson: e1_new = alpha × e1_old + strain_coeff × (strain_old + strain_new)
                        e1(ix, iy, iz, ei, uu) = alpha * e1(ix, iy, iz, ei, uu)
                                                 + sc * (strain_old_val + strain_new);
                    }   
                    // Save strain for next timestep
                    dot_e1_old(ix, iy, iz, ei) = strain_new;
                }
            }
        }
    });  // End of forall_3D
}




// =============================================================================
// Explicit Template Instantiations
// =============================================================================

#define INSTANTIATE_VISCO_ACOUSTIC_3D(NGLL) \
    template void SEMVisco_IsotropicAcousticIntegrator3D::AddMultPA_Visco_Opt<NGLL>(const Vector&, Vector&) const;

SEM_INSTANTIATE_NGLL_3D(INSTANTIATE_VISCO_ACOUSTIC_3D)

#undef INSTANTIATE_VISCO_ACOUSTIC_3D

}  // namespace SEM
