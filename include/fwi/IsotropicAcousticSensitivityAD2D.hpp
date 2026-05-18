/**
 * @file IsotropicAcousticSensitivityAD2D.hpp
 * @brief AD-based sensitivity kernel for 2D isotropic acoustic media.
 *
 * Forward-mode automatic differentiation via mfem::future::dual replaces the
 * hand-written chain-rule expressions in IsotropicAcousticSensitivity2D.
 *
 * The public API matches the hand-written class exactly: Accumulate, Save,
 * Reset, SaveHessian, ResetHessian, Ngll, VpKernel, RhoKernel. This is
 * intentional — the SensitivityKernelFactory swaps the backend at runtime
 * based on the YAML key `inversion.sensitivity.backend: [hand|ad]`.
 *
 * Internal layout (per-GLL-point):
 *   - Stiffness path (ρ kernel): seed dual over inv_rho, call
 *     AcousticFluxPhysical2D<dual>, contract with ∇λ, accumulate into
 *     `k_invrho_` (∂J/∂(1/ρ) accumulator).
 *   - Mass path (Vp kernel): seed dual over inv_kappa, compute
 *     `(1/κ)·p̈` as dual, multiply by p_adj, accumulate into
 *     `k_invkappa_` (∂J/∂(1/κ) accumulator).
 *   - Save(): convert from (K_{1/ρ}, K_{1/κ}) to (K_ρ, K_Vp) via the
 *     standard (Vp, ρ)-independent chain rule:
 *         K_Vp  =  -2/(ρ·Vp³)  ·  K_{1/κ}        (mass path only)
 *         K_ρ   =  -1/ρ²       ·  K_{1/ρ}        (stiffness path only)
 *
 * References:
 *   - include/fwi/IsotropicAcousticSensitivity2D.hpp    (hand version)
 *   - include/integ/kernels/AcousticFluxKernel2D.hpp    (shared template)
 */

#ifndef SEM_ISOTROPIC_ACOUSTIC_SENSITIVITY_AD_2D_HPP
#define SEM_ISOTROPIC_ACOUSTIC_SENSITIVITY_AD_2D_HPP

#include "fwi/SensitivityKernel.hpp"
#include "integ/core/SEMGeometry.hpp"
#include "integ/core/SEMDofOrdering.hpp"
#include "material/MaterialField.hpp"

namespace SEM {

using mfem::ParFiniteElementSpace;

class IsotropicAcousticSensitivityAD2D : public SensitivityKernelBase2D {
public:
    /**
     * @brief Construct AD-based sensitivity kernel (same signature as hand version).
     *
     * @param fes Scalar finite element space (same as forward simulation)
     * @param kappa Bulk modulus field (κ at GLL points)
     * @param inv_rho Inverse density field (1/ρ at GLL points)
     */
    /// @param unrelaxed_correction Per-GLL c_i = κ_u/κ_user field from
    ///                              viscoacoustic material; pass nullptr for
    ///                              pure acoustic. FinalizeKernels chain-rule
    ///                              recovers K_{Vp_user} from K_{1/κ_u}.
    IsotropicAcousticSensitivityAD2D(ParFiniteElementSpace& fes,
                                      const MaterialField& kappa,
                                      const MaterialField& inv_rho,
                                      const MaterialField* unrelaxed_correction = nullptr);

    /// Cache material fields for device access (separate from ctor for CUDA)
    void InitMaterialFields(const MaterialField& kappa,
                            const MaterialField& inv_rho,
                            const MaterialField* unrelaxed_correction = nullptr);

    void Accumulate(const Vector& fwd_p, const Vector& fwd_a,
                    const Vector& adj_p, real_t dt) override;

    void Save(const std::string& dir, ParMesh& mesh, int source_id) override;
    void Reset() override;
    int Ngll() const override { return ngll_; }

    void SaveHessian(const std::string& dir, ParMesh& mesh, int source_id) override;
    void ResetHessian() override;

    /// Apply the chain rule to populate vp_kernel_ and rho_kernel_
    /// from the raw (k_invrho_, k_invkappa_) accumulators. Called
    /// automatically by Save() and by VpKernel()/RhoKernel(); tests may
    /// also call it explicitly to compare against the hand version.
    void FinalizeKernels() const;

    const Vector& VpKernel()  const { FinalizeKernels(); return vp_kernel_;  }
    const Vector& RhoKernel() const { FinalizeKernels(); return rho_kernel_; }

    // GPU kernel methods exposed for CUDA extended-lambda instantiation.
    template<int NGLL>
    void AccumulateRhoKernel_AD(const Vector& fwd_p, const Vector& adj_p, real_t dt);

    template<int NGLL>
    void AccumulateVpKernel_AD(const Vector& fwd_a, const Vector& adj_p, real_t dt);

private:
    SEMGeometry2D geom_;
    SEMDofOrdering2D dofs_;

    // Material fields as per-GLL-point scalars (GPU-resident views).
    Vector inv_rho_;      // 1/ρ at GLL points
    Vector inv_kappa_;    // 1/κ at GLL points
    Vector kappa_;        // κ at GLL points (stored raw to avoid 1/(1/κ) roundoff)
    Vector unrelaxed_correction_;   // c_i per GLL (1 for pure acoustic)

    // AD accumulators (raw derivatives in seed-parameter space).
    Vector k_invrho_;     // Σ ∂J/∂(1/ρ) contributions × dt
    Vector k_invkappa_;   // Σ ∂J/∂(1/κ) contributions × dt

    // Pseudo-Hessian accumulators (still computed analytically; same as hand).
    Vector vp_hessian_;   // Σ 4/(ρ²·Vp⁶)·p̈² · dt
    Vector rho_hessian_;  // Σ (1/ρ²)·|∇φ_fwd|² · dt

    // Public-API kernels in (Vp, ρ) space, derived from k_invrho_/k_invkappa_
    // at Save() time via the (Vp, ρ)-independent chain rule.
    mutable Vector vp_kernel_;
    mutable Vector rho_kernel_;

    ParFiniteElementSpace* fes_;
    int ngll_ = 0;
    int ne_ = 0;
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ACOUSTIC_SENSITIVITY_AD_2D_HPP
