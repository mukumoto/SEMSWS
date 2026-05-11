/**
 * @file OperatorFactory.cpp
 * @brief Implementation of operator factory functions
 */

#include "operator/OperatorFactory.hpp"
#include "util/Profiler.hpp"

namespace SEM {

// =============================================================================
// Type-safe 2D factory functions
// =============================================================================

std::unique_ptr<WaveOperator> CreateElasticOperator2D(
    ParFiniteElementSpace& fes,
    int order,
    const ElasticMaterialBase2D& material,
    const OperatorConfig& config)
{
    auto elastic_op = std::make_unique<ElasticOperator2D>(fes, order, material);

    // Setup mass matrix
    elastic_op->SetupMass();

    // Setup Cerjan taper if configured
    if (config.HasDamping()) {
        elastic_op->SetupCerjanTaper(config.damping);
    }

    // CRITICAL: Set dt for viscoelastic integrator BEFORE SetupStiffness
    // EnableAttenuation() needs dt to compute Crank-Nicolson coefficients.
    // Without this, dt_=0 causes all viscoelastic effects to be zero.
    elastic_op->SetTimeStep(config.dt);

    // Setup stiffness (creates viscoelastic integrator if material.HasAttenuation())
    elastic_op->SetupStiffness();

    return elastic_op;
}

std::unique_ptr<WaveOperator> CreateAcousticOperator2D(
    ParFiniteElementSpace& fes,
    int order,
    const AcousticMaterialBase2D& material,
    const OperatorConfig& config)
{
    auto acoustic_op = std::make_unique<AcousticOperator2D>(fes, order, material);

    // Setup mass matrix
    // Note: For viscoacoustic, material.Kappa() is already the unrelaxed value
    // (corrected in IsotropicAcousticMaterial::ApplyAttenuationCorrection())
    acoustic_op->SetupMass();

    // Setup Cerjan taper if configured
    if (config.HasDamping()) {
        acoustic_op->SetupCerjanTaper(config.damping);
    }

    // CRITICAL: Set dt for viscoacoustic integrator BEFORE SetupStiffness
    acoustic_op->SetTimeStep(config.dt);

    // Setup stiffness (creates viscoacoustic integrator if material.HasAttenuation())
    acoustic_op->SetupStiffness();

    // Setup Dirichlet BC if configured
    if (config.dirichlet_tdof.Size() > 0) {
        acoustic_op->SetupDirichletBC(config.dirichlet_tdof);
        acoustic_op->SetDirichletLDofs(config.dirichlet_ldof);
    }

    return acoustic_op;
}

// =============================================================================
// Type-safe 3D factory functions
// =============================================================================

std::unique_ptr<WaveOperator> CreateElasticOperator3D(
    ParFiniteElementSpace& fes,
    int order,
    const ElasticMaterialBase3D& material,
    const OperatorConfig& config)
{
    std::unique_ptr<ElasticOperator3D> elastic_op;
    {
        PROFILE_REGION("Setup:CreateElasticOp3D");
        elastic_op = std::make_unique<ElasticOperator3D>(fes, order, material);
    }

    {
        PROFILE_REGION("Setup:SetupMass3D");
        elastic_op->SetupMass();
    }

    // Setup Cerjan taper if configured
    if (config.HasDamping()) {
        PROFILE_REGION("Setup:SetupCerjanTaper3D");
        elastic_op->SetupCerjanTaper(config.damping);
    }

    // CRITICAL: Set dt for viscoelastic integrator BEFORE SetupStiffness
    elastic_op->SetTimeStep(config.dt);

    // Setup stiffness (creates viscoelastic integrator if material.HasAttenuation())
    {
        PROFILE_REGION("Setup:SetupStiffness3D");
        elastic_op->SetupStiffness();
    }

    return elastic_op;
}

std::unique_ptr<WaveOperator> CreateAcousticOperator3D(
    ParFiniteElementSpace& fes,
    int order,
    const AcousticMaterialBase3D& material,
    const OperatorConfig& config)
{
    auto acoustic_op = std::make_unique<AcousticOperator3D>(fes, order, material);

    // Setup mass matrix
    acoustic_op->SetupMass();

    // Setup Cerjan taper if configured
    if (config.HasDamping()) {
        acoustic_op->SetupCerjanTaper(config.damping);
    }

    // CRITICAL: Set dt for viscoacoustic integrator BEFORE SetupStiffness
    // EnableAttenuation() needs dt for Crank-Nicolson coefficients.
    // Without this, dt_=0 causes all viscoacoustic effects to be zero.
    acoustic_op->SetTimeStep(config.dt);

    // Setup stiffness (creates viscoacoustic integrator if material.HasAttenuation())
    acoustic_op->SetupStiffness();

    // Setup Dirichlet BC if configured
    if (config.dirichlet_tdof.Size() > 0) {
        acoustic_op->SetupDirichletBC(config.dirichlet_tdof);
        acoustic_op->SetDirichletLDofs(config.dirichlet_ldof);
    }

    return acoustic_op;
}

}  // namespace SEM
