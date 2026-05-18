/**
 * @file IsotropicElasticSensitivity2D.hpp
 * @brief Sensitivity kernel for 2D isotropic (plane-strain) elastic media
 *
 * Accumulates Vp, Vs, and ρ sensitivity kernels from a displacement-
 * formulation forward wavefield u = (ux, uy) and its adjoint λ^*.
 *
 * Starting kernels in Lamé parameters (λ, μ, ρ) are classical:
 *   K_λ = -∫ (∇·u)(∇·λ^*) dt
 *   K_μ = -∫ 2 ε(u):ε(λ^*) dt        with ε:ε' = Σ_ij ε_ij·ε'_ij
 *   K_ρ = -∫ ü · λ^* dt              (vector dot)
 *
 * These are converted to (Vp_user, Vs_user, ρ) at Save() time via the
 * standard chain rule.  With viscoelastic attenuation the forward solver
 * internally uses unrelaxed moduli (λ_u, μ_u) = (c_κ κ_user − c_μ μ_user,
 * c_μ μ_user) in 2D (plane-strain κ = λ + μ). Propagating a perturbation
 * of the user-facing (Vp, Vs) through the chain rule gives:
 *   K_Vp = 2 c_κ ρ Vp · K_λu
 *   K_Vs = 2 c_μ ρ Vs · K_μu − 2 (c_κ + c_μ) ρ Vs · K_λu
 *   K_ρ  = K_ρ                                  (untransformed)
 * Pure elastic (c_κ = c_μ = 1) reduces to the classical forms
 *   K_Vp = 2 ρ Vp · K_λ,  K_Vs = 2 ρ Vs · (K_μ − 2 K_λ).
 *
 * Pseudo-Hessian (Shin diagonal) uses the sum of squared chain-rule
 * coefficients:
 *   H_Vp = (2 c_κ ρ Vp)² · H_λu
 *   H_Vs = (2 c_μ ρ Vs)² · H_μu + (2 (c_κ+c_μ) ρ Vs)² · H_λu
 *   H_ρ  = ∫ |ü_fwd|² dt
 */

#ifndef SEM_ISOTROPIC_ELASTIC_SENSITIVITY_2D_HPP
#define SEM_ISOTROPIC_ELASTIC_SENSITIVITY_2D_HPP

#include "fwi/SensitivityKernel.hpp"
#include "integ/core/SEMGeometry.hpp"
#include "integ/core/SEMDofOrdering.hpp"
#include "material/MaterialField.hpp"

namespace SEM {

using mfem::ParFiniteElementSpace;

class IsotropicElasticSensitivity2D : public SensitivityKernelBase2D {
public:
    /**
     * @param fes    Vector (vdim=2) FE space used by the elastic forward.
     * @param lambda Lamé first parameter per GLL point.
     * @param mu     Shear modulus per GLL point.
     * @param rho    Density per GLL point.
     */
    /// @param c_kappa  Per-GLL c_κ = κ_u/κ_user (viscoelastic; nullptr ⇒ 1).
    /// @param c_mu     Per-GLL c_μ = μ_u/μ_user (viscoelastic; nullptr ⇒ 1).
    IsotropicElasticSensitivity2D(ParFiniteElementSpace& fes,
                                   const MaterialField& lambda,
                                   const MaterialField& mu,
                                   const MaterialField& rho,
                                   const MaterialField* c_kappa = nullptr,
                                   const MaterialField* c_mu    = nullptr);

    /// Refresh cached chain-rule coefficients for (λ_u, μ_u, ρ) → (Vp_user, Vs_user, ρ).
    void InitCoefficients(const MaterialField& lambda,
                          const MaterialField& mu,
                          const MaterialField& rho,
                          const MaterialField* c_kappa = nullptr,
                          const MaterialField* c_mu    = nullptr);

    void Accumulate(const Vector& fwd_u, const Vector& fwd_a,
                    const Vector& adj_u, real_t dt) override;

    void Save(const std::string& dir, ParMesh& mesh, int source_id) override;
    void SaveHessian(const std::string& dir, ParMesh& mesh, int source_id) override;
    void Reset() override;
    void ResetHessian() override;
    int Ngll() const override { return ngll_; }

    const Vector& LambdaKernel() const { return k_lambda_; }
    const Vector& MuKernel()     const { return k_mu_; }
    const Vector& RhoKernel()    const { return k_rho_; }

    // GPU kernel (public for CUDA extended lambda)
    template <int NGLL>
    void AccumulateKernel(const Vector& fwd_u, const Vector& fwd_a,
                          const Vector& adj_u, real_t dt);

private:
    SEMGeometry2D geom_;
    SEMDofOrdering2D dofs_;

    // Save()-time chain-rule coefficients.
    //   K_Vp = coeff_vp · K_λu
    //   K_Vs = coeff_vs_mu · K_μu + coeff_vs_lam · K_λu
    // Pure elastic (c_κ=c_μ=1):
    //   coeff_vp     = 2 ρ Vp
    //   coeff_vs_mu  = 2 ρ Vs
    //   coeff_vs_lam = -4 ρ Vs   → K_Vs = 2 ρ Vs (K_μ − 2 K_λ).
    // Viscoelastic 2D (κ = λ+μ, plane-strain):
    //   coeff_vp     = 2 c_κ ρ Vp_user
    //   coeff_vs_mu  = 2 c_μ ρ Vs_user
    //   coeff_vs_lam = -2 (c_κ + c_μ) ρ Vs_user
    Vector rho_;
    Vector coeff_vp_;
    Vector coeff_vs_mu_;
    Vector coeff_vs_lam_;

    // Accumulated kernels in (λ, μ, ρ) space.
    Vector k_lambda_;
    Vector k_mu_;
    Vector k_rho_;

    // Accumulated pseudo-Hessian diagonals (forward-only illumination).
    Vector h_lambda_;     // Σ (∇·u_fwd)² dt
    Vector h_mu_;         // Σ 2 ε(u_fwd):ε(u_fwd) dt
    Vector h_rho_;        // Σ |ü_fwd|² dt

    ParFiniteElementSpace* fes_;
    int ngll_ = 0;
    int ne_   = 0;
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ELASTIC_SENSITIVITY_2D_HPP
