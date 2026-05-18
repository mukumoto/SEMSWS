/**
 * @file IsotropicElasticSensitivityAD3D.hpp
 * @brief Forward-mode AD sensitivity kernel for 3D isotropic elastic media.
 *
 * 3D counterpart to IsotropicElasticSensitivityAD2D. Same two single-seed
 * AD passes per quadrature point (λ, then μ) + direct ρ path, extended to
 * the full 3D strain-stress contraction. Save() applies the chain rule
 * (identical to the 2D form).
 */

#ifndef SEM_ISOTROPIC_ELASTIC_SENSITIVITY_AD_3D_HPP
#define SEM_ISOTROPIC_ELASTIC_SENSITIVITY_AD_3D_HPP

#include "fwi/SensitivityKernel.hpp"
#include "integ/core/SEMGeometry.hpp"
#include "integ/core/SEMDofOrdering.hpp"
#include "material/MaterialField.hpp"

namespace SEM {

using mfem::ParFiniteElementSpace;

class IsotropicElasticSensitivityAD3D : public SensitivityKernelBase3D {
public:
    IsotropicElasticSensitivityAD3D(ParFiniteElementSpace& fes,
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

    Vector lambda_;
    Vector mu_;
    // 3D chain-rule coefficients (see IsotropicElasticSensitivity3D doc).
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

#endif  // SEM_ISOTROPIC_ELASTIC_SENSITIVITY_AD_3D_HPP
