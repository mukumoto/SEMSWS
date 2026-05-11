/**
 * @file ElasticityOperator3D.hpp
 * @brief 3D elastic wave operator (isotropic, extensible to anisotropic)
 */

#ifndef SEM_OPERATOR_ELASTIC_ELASTICITY_OPERATOR_3D_HPP
#define SEM_OPERATOR_ELASTIC_ELASTICITY_OPERATOR_3D_HPP

#include "operator/WaveOperatorBase3D.hpp"
#include "material/ElasticMaterialBase.hpp"

namespace SEM {

/**
 * @class ElasticOperator3D
 * @brief 3D elastic wave operator
 *
 * Solves: rho * d2u/dt2 = div(sigma) + f
 *
 * Supports multiple material types via GetType() dispatch:
 * - Isotropic: sigma = lambda * tr(epsilon) * I + 2 * mu * epsilon
 * - Anisotropic: (future) sigma_ij = C_ijkl * epsilon_kl
 * - VTI: (future) transversely isotropic
 *
 * Uses diagonal mass matrix (GLL quadrature) for explicit time stepping.
 */
class ElasticOperator3D : public WaveOperatorBase3D {
public:
    /**
     * @brief Construct elasticity operator
     * @param fes Finite element space (vector H1, dim=3)
     * @param order Polynomial order
     * @param material Material model (any ElasticMaterialBase3D subclass)
     */
    ElasticOperator3D(
        ParFiniteElementSpace& fes,
        int order,
        const ElasticMaterialBase3D& material);

    ~ElasticOperator3D() override = default;

    // Builder methods (return *this for chaining)
    ElasticOperator3D& SetupMass() override;
    ElasticOperator3D& SetupStiffness() override;

    bool HasViscoelasticity() const override;
    void PrintInfo(std::ostream& os = mfem::out) const override;

    // Viscoelastic memory state I/O for Revolve checkpointing (3D:
    // 6 symmetric stress components per SLS × strain_old).
    int AttenuationStateSize() const override;
    void GetAttenuationState(Vector& state) const override;
    void SetAttenuationState(const Vector& state) override;

protected:
    void ResetPhysicsState() override;

private:
    const ElasticMaterialBase3D& material_;
};

}  // namespace SEM

#endif  // SEM_OPERATOR_ELASTIC_ELASTICITY_OPERATOR_3D_HPP
