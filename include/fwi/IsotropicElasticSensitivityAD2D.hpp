/**
 * @file IsotropicElasticSensitivityAD2D.hpp
 * @brief Forward-mode AD sensitivity kernel for 2D isotropic elastic media.
 *
 * Drop-in replacement for IsotropicElasticSensitivity2D — same public API,
 * same (K_Vp, K_Vs, K_ρ) output, same pseudo-Hessian, but the material
 * tangent is computed by `mfem::future::dual<real_t, real_t>` applied to
 * the shared `ElasticStressPhysical2D<T>` template. Two single-seed AD
 * passes per quadrature point (one for λ, one for μ) keep GPU register
 * pressure comparable to acoustic AD, and the ρ path is pointwise (no AD
 * needed).
 *
 * Intended use: runtime-switchable via `inversion.sensitivity.backend: ad`,
 * with the hand version serving as the golden reference.
 */

#ifndef SEM_ISOTROPIC_ELASTIC_SENSITIVITY_AD_2D_HPP
#define SEM_ISOTROPIC_ELASTIC_SENSITIVITY_AD_2D_HPP

#include "fwi/SensitivityKernel.hpp"
#include "integ/core/SEMGeometry.hpp"
#include "integ/core/SEMDofOrdering.hpp"
#include "material/MaterialField.hpp"

namespace SEM {

using mfem::ParFiniteElementSpace;

class IsotropicElasticSensitivityAD2D : public SensitivityKernelBase2D {
public:
    IsotropicElasticSensitivityAD2D(ParFiniteElementSpace& fes,
                                     const MaterialField& lambda,
                                     const MaterialField& mu,
                                     const MaterialField& rho,
                                     const MaterialField* c_kappa = nullptr,
                                     const MaterialField* c_mu    = nullptr);

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

    template <int NGLL>
    void AccumulateKernel(const Vector& fwd_u, const Vector& fwd_a,
                          const Vector& adj_u, real_t dt);

private:
    SEMGeometry2D geom_;
    SEMDofOrdering2D dofs_;

    // Material fields kept verbatim so the dual passes can seed λ_u and μ_u
    // directly. Save() uses the (c_κ, c_μ)-aware chain-rule coefficients
    // (see IsotropicElasticSensitivity2D for the math).
    Vector lambda_;
    Vector mu_;
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

#endif  // SEM_ISOTROPIC_ELASTIC_SENSITIVITY_AD_2D_HPP
