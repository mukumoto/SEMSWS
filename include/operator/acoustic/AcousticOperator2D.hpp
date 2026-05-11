/**
 * @file AcousticOperator2D.hpp
 * @brief 2D acoustic wave operator
 */

#ifndef SEM_OPERATOR_ACOUSTIC_ACOUSTIC_OPERATOR_2D_HPP
#define SEM_OPERATOR_ACOUSTIC_ACOUSTIC_OPERATOR_2D_HPP

#include "operator/WaveOperatorBase2D.hpp"
#include "material/AcousticMaterialBase.hpp"

namespace SEM {

/**
 * @class AcousticOperator2D
 * @brief 2D acoustic wave operator
 *
 * Solves: (1/kappa) * d2phi/dt2 = div(1/rho * grad(phi)) + f
 * where phi is the displacement potential, kappa is bulk modulus
 *
 * Supports viscoacoustic (attenuation) via Generalized Zener (SLS) model.
 * When attenuation is enabled, uses Newmark scheme.
 *
 * Uses diagonal mass matrix (GLL quadrature) for explicit time stepping.
 */
class AcousticOperator2D : public WaveOperatorBase2D {
public:
    /**
     * @brief Construct acoustic operator
     * @param fes Finite element space (scalar H1)
     * @param order Polynomial order
     * @param material Acoustic material (kappa, 1/rho)
     */
    AcousticOperator2D(
        ParFiniteElementSpace& fes,
        int order,
        const AcousticMaterialBase2D& material);

    ~AcousticOperator2D() override = default;

    // Builder methods
    AcousticOperator2D& SetupMass() override;
    AcousticOperator2D& SetupStiffness() override;
    AcousticOperator2D& SetupDirichletBC(const Array<int>& tdof_list) override;
    void SetDirichletLDofs(const Array<int>& ldof_list);

    /// Enforce Dirichlet BC: set chi=chi_dot=chi_ddot=0 at boundary nodes
    /// Zero-pressure Dirichlet boundary condition
    void EnforceDirichletBC(ParGridFunction& u,
                            ParGridFunction& dudt,
                            ParGridFunction& dudt2) override;

    bool HasViscoelasticity() const override;
    void PrintInfo(std::ostream& os = mfem::out) const override;

    // Checkpointing: attenuation state pack/unpack
    int AttenuationStateSize() const override;
    void GetAttenuationState(Vector& state) const override;
    void SetAttenuationState(const Vector& state) override;

protected:
    void ApplyBoundaryConditions() override;
    void ResetPhysicsState() override;

private:
    const AcousticMaterialBase2D& material_;

    // Dirichlet BC DOF lists
    Array<int> dirichlet_tdof_;  // true DOFs (for RHS)
    Array<int> dirichlet_ldof_;  // local DOFs (for state vectors)
};

}  // namespace SEM

#endif  // SEM_OPERATOR_ACOUSTIC_ACOUSTIC_OPERATOR_2D_HPP
