/**
 * @file ElasticityOperator2D.hpp
 * @brief 2D elastic wave operator (isotropic, extensible to anisotropic)
 */

#ifndef SEM_OPERATOR_ELASTIC_ELASTICITY_OPERATOR_2D_HPP
#define SEM_OPERATOR_ELASTIC_ELASTICITY_OPERATOR_2D_HPP

#include "operator/WaveOperatorBase2D.hpp"
#include "material/ElasticMaterialBase.hpp"

namespace SEM {

/**
 * @class ElasticOperator2D
 * @brief 2D elastic wave operator
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
class ElasticOperator2D : public WaveOperatorBase2D {
public:
    /**
     * @brief Construct elasticity operator
     * @param fes Finite element space (vector H1, dim=2)
     * @param order Polynomial order
     * @param material Material model (any ElasticMaterialBase2D subclass)
     */
    ElasticOperator2D(
        ParFiniteElementSpace& fes,
        int order,
        const ElasticMaterialBase2D& material);

    ~ElasticOperator2D() override = default;

    // Builder methods (return *this for chaining)
    ElasticOperator2D& SetupMass() override;
    ElasticOperator2D& SetupStiffness() override;

    bool HasViscoelasticity() const override;
    void PrintInfo(std::ostream& os = mfem::out) const override;

    // Viscoelastic memory state I/O for Revolve checkpointing.
    // Packs (M_packed, strain_old_packed) as a flat Vector.
    int AttenuationStateSize() const override;
    void GetAttenuationState(Vector& state) const override;
    void SetAttenuationState(const Vector& state) override;

protected:
    void ResetPhysicsState() override;

private:
    const ElasticMaterialBase2D& material_;
};

}  // namespace SEM

#endif  // SEM_OPERATOR_ELASTIC_ELASTICITY_OPERATOR_2D_HPP
