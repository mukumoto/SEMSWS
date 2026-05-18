/**
 * @file IsotropicElasticSensitivity3D.hpp
 * @brief Sensitivity kernel for 3D isotropic elastic media.
 *
 * Dim=3 counterpart to IsotropicElasticSensitivity2D. Identical math form,
 * extended to 3 spatial directions and 6 strain components.
 *
 *   K_λ = -∫ (∇·u)(∇·λ*) dt            (∇·u = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z)
 *   K_μ = -∫ 2 ε(u):ε(λ*) dt           (full 3×3 symmetric contraction)
 *   K_ρ = -∫ ü · λ* dt                 (3-component vector dot)
 *
 * Chain rule in Save() (same as 2D):
 *   K_Vp = 2ρVp · K_λ
 *   K_Vs = 2ρVs · (K_μ − 2 K_λ)
 *   K_ρ  = K_ρ
 */

#ifndef SEM_ISOTROPIC_ELASTIC_SENSITIVITY_3D_HPP
#define SEM_ISOTROPIC_ELASTIC_SENSITIVITY_3D_HPP

#include "fwi/SensitivityKernel.hpp"
#include "integ/core/SEMGeometry.hpp"
#include "integ/core/SEMDofOrdering.hpp"
#include "material/MaterialField.hpp"

namespace SEM {

using mfem::ParFiniteElementSpace;

class IsotropicElasticSensitivity3D : public SensitivityKernelBase3D {
public:
    IsotropicElasticSensitivity3D(ParFiniteElementSpace& fes,
                                   const MaterialField3D& lambda,
                                   const MaterialField3D& mu,
                                   const MaterialField3D& rho,
                                   const MaterialField3D* c_kappa = nullptr,
                                   const MaterialField3D* c_mu    = nullptr);

    void InitCoefficients(const MaterialField3D& lambda,
                          const MaterialField3D& mu,
                          const MaterialField3D& rho,
                          const MaterialField3D* c_kappa = nullptr,
                          const MaterialField3D* c_mu    = nullptr);

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

    template <int NGLL>
    void AccumulateKernel(const Vector& fwd_u, const Vector& fwd_a,
                          const Vector& adj_u, real_t dt);

private:
    SEMGeometry3D geom_;
    SEMDofOrdering3D dofs_;

    // Save()-time chain-rule coefficients (c_κ-, c_μ-aware; see 2D doc).
    //   3D: κ = λ + 2μ/3, so
    //   coeff_vp     = 2 c_κ ρ Vp_user
    //   coeff_vs_mu  = 2 c_μ ρ Vs_user
    //   coeff_vs_lam = −(4/3) (2 c_κ + c_μ) ρ Vs_user
    Vector rho_;
    Vector coeff_vp_;
    Vector coeff_vs_mu_;
    Vector coeff_vs_lam_;

    Vector k_lambda_, k_mu_, k_rho_;
    Vector h_lambda_, h_mu_, h_rho_;

    ParFiniteElementSpace* fes_;
    int ngll_ = 0;
    int ne_   = 0;
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ELASTIC_SENSITIVITY_3D_HPP
