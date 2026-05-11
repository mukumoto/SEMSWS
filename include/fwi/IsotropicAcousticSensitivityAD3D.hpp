/**
 * @file IsotropicAcousticSensitivityAD3D.hpp
 * @brief AD-based sensitivity kernel for 3D isotropic acoustic media.
 *
 * 3D counterpart to IsotropicAcousticSensitivityAD2D. Same two-seed layout
 * (inv_rho for the stiffness path, inv_kappa for the mass path) with
 * forward-mode AD via mfem::future::dual; FinalizeKernels() applies the
 * TOY2DAC chain rule to populate (K_Vp, K_ρ) at Save() time.
 */

#ifndef SEM_ISOTROPIC_ACOUSTIC_SENSITIVITY_AD_3D_HPP
#define SEM_ISOTROPIC_ACOUSTIC_SENSITIVITY_AD_3D_HPP

#include "fwi/SensitivityKernel.hpp"
#include "integ/core/SEMGeometry.hpp"
#include "integ/core/SEMDofOrdering.hpp"
#include "material/MaterialField.hpp"

namespace SEM {

using mfem::ParFiniteElementSpace;

class IsotropicAcousticSensitivityAD3D : public SensitivityKernelBase3D {
public:
    /// @param unrelaxed_correction Per-GLL c_i = κ_u/κ_user field from
    ///                              viscoacoustic material; pass nullptr for
    ///                              pure acoustic. FinalizeKernels chain-rule
    ///                              recovers K_{Vp_user} from K_{1/κ_u}.
    IsotropicAcousticSensitivityAD3D(ParFiniteElementSpace& fes,
                                      const MaterialField3D& kappa,
                                      const MaterialField3D& inv_rho,
                                      const MaterialField3D* unrelaxed_correction = nullptr);

    void InitMaterialFields(const MaterialField3D& kappa,
                            const MaterialField3D& inv_rho,
                            const MaterialField3D* unrelaxed_correction = nullptr);

    void Accumulate(const Vector& fwd_p, const Vector& fwd_a,
                    const Vector& adj_p, real_t dt) override;

    void Save(const std::string& dir, ParMesh& mesh, int source_id) override;
    void Reset() override;
    int Ngll() const override { return ngll_; }

    void SaveHessian(const std::string& dir, ParMesh& mesh, int source_id) override;
    void ResetHessian() override;

    void FinalizeKernels() const;

    const Vector& VpKernel()  const { FinalizeKernels(); return vp_kernel_;  }
    const Vector& RhoKernel() const { FinalizeKernels(); return rho_kernel_; }

    template <int NGLL>
    void AccumulateRhoKernel_AD(const Vector& fwd_p, const Vector& adj_p, real_t dt);

    template <int NGLL>
    void AccumulateVpKernel_AD(const Vector& fwd_a, const Vector& adj_p, real_t dt);

private:
    SEMGeometry3D geom_;
    SEMDofOrdering3D dofs_;

    Vector inv_rho_;
    Vector inv_kappa_;
    Vector kappa_;
    Vector unrelaxed_correction_;   // c_i per GLL (1 for pure acoustic)

    Vector k_invrho_;
    Vector k_invkappa_;

    Vector vp_hessian_;
    Vector rho_hessian_;

    mutable Vector vp_kernel_;
    mutable Vector rho_kernel_;

    ParFiniteElementSpace* fes_;
    int ngll_ = 0;
    int ne_   = 0;
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ACOUSTIC_SENSITIVITY_AD_3D_HPP
