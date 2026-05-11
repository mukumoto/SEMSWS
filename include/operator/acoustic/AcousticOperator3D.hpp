/**
 * @file AcousticOperator3D.hpp
 * @brief 3D acoustic wave operator
 */

#ifndef SEM_OPERATOR_ACOUSTIC_ACOUSTIC_OPERATOR_3D_HPP
#define SEM_OPERATOR_ACOUSTIC_ACOUSTIC_OPERATOR_3D_HPP

#include "operator/WaveOperatorBase3D.hpp"
#include "material/AcousticMaterialBase.hpp"

namespace SEM {

/**
 * @class AcousticOperator3D
 * @brief 3D acoustic wave operator
 *
 * Solves: (1/kappa) * d2phi/dt2 = div(1/rho * grad(phi)) + f
 * where phi is the displacement potential, kappa is bulk modulus
 *
 * Uses diagonal mass matrix (GLL quadrature) for explicit time stepping.
 */
class AcousticOperator3D : public WaveOperatorBase3D {
public:
    /**
     * @brief Construct acoustic operator
     * @param fes Finite element space (scalar H1)
     * @param order Polynomial order
     * @param material Acoustic material (kappa, 1/rho)
     */
    AcousticOperator3D(
        ParFiniteElementSpace& fes,
        int order,
        const AcousticMaterialBase3D& material);

    ~AcousticOperator3D() override = default;

    // Builder methods
    AcousticOperator3D& SetupMass() override;
    AcousticOperator3D& SetupStiffness() override;
    AcousticOperator3D& SetupDirichletBC(const Array<int>& tdof_list) override;
    void SetDirichletLDofs(const Array<int>& ldof_list);

    /// Enforce Dirichlet BC: set chi=chi_dot=chi_ddot=0 at boundary nodes
    void EnforceDirichletBC(ParGridFunction& u,
                            ParGridFunction& dudt,
                            ParGridFunction& dudt2) override;

    void PrintInfo(std::ostream& os = mfem::out) const override;

    /// True iff the material has attenuation (viscoacoustic).
    bool HasViscoelasticity() const override;

    // Checkpointing: attenuation state pack/unpack (visco-acoustic memory
    // variables e1, dot_e1_old). Required for FWI adjoint checkpoint/restore.
    int AttenuationStateSize() const override;
    void GetAttenuationState(Vector& state) const override;
    void SetAttenuationState(const Vector& state) override;

protected:
    /// Reset viscoacoustic memory (called between sources in FWI).
    void ResetPhysicsState() override;

    void ApplyBoundaryConditions() override;

private:
    const AcousticMaterialBase3D& material_;

    // Dirichlet BC DOF lists
    Array<int> dirichlet_tdof_;  // true DOFs (for RHS)
    Array<int> dirichlet_ldof_;  // local DOFs (for state vectors)
};

}  // namespace SEM

#endif  // SEM_OPERATOR_ACOUSTIC_ACOUSTIC_OPERATOR_3D_HPP
