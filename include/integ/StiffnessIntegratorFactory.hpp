/**
 * @file StiffnessIntegratorFactory.hpp
 * @brief Factory functions that construct the stiffness integrator matched
 *        to a given (domain, dimension, material) combination.
 *
 * Four entry points, one per (physics, dim) pair. Each is narrow: the caller
 * (an Operator) already knows the physics and dimension from its own type,
 * so the factory only dispatches on the concrete material class (Isotropic
 * vs. Anisotropic, pure vs. visco, ...).
 *
 * Rationale: keeping material.hpp free of any integrator types avoids
 * pulling sem_integ headers into sem_material, and keeps the layering
 * one-way (sem_integ depends on sem_material, never the reverse). No
 * cross-library virtual dispatch is required.
 */

#ifndef SEM_STIFFNESS_INTEGRATOR_FACTORY_HPP
#define SEM_STIFFNESS_INTEGRATOR_FACTORY_HPP

#include <mfem.hpp>
#include <memory>

namespace SEM {

class ElasticMaterialBase2D;
class ElasticMaterialBase3D;
class AcousticMaterialBase2D;
class AcousticMaterialBase3D;

using mfem::real_t;

/// Build the 2D elastic stiffness integrator for `material`.
/// The returned integrator has already had AssemblePA(fes) called (and, for
/// viscoelastic materials, EnableAttenuation + FinalizeMaterialParams).
///
/// @param material  Concrete 2D elastic material (isotropic today; anisotropic later)
/// @param fes       Vector-valued FE space (vdim = 2)
/// @param dt        Time step (required for viscoelastic Crank-Nicolson coefs)
std::unique_ptr<mfem::BilinearFormIntegrator>
CreateElasticStiffnessIntegrator2D(const ElasticMaterialBase2D& material,
                                   mfem::ParFiniteElementSpace& fes,
                                   real_t dt);

/// Build the 3D elastic stiffness integrator for `material`.
std::unique_ptr<mfem::BilinearFormIntegrator>
CreateElasticStiffnessIntegrator3D(const ElasticMaterialBase3D& material,
                                   mfem::ParFiniteElementSpace& fes,
                                   real_t dt);

/// Build the 2D acoustic stiffness integrator for `material`.
/// Handles both pure acoustic and viscoacoustic (based on HasAttenuation()).
std::unique_ptr<mfem::BilinearFormIntegrator>
CreateAcousticStiffnessIntegrator2D(const AcousticMaterialBase2D& material,
                                    mfem::ParFiniteElementSpace& fes,
                                    real_t dt);

/// Build the 3D acoustic stiffness integrator for `material`.
/// 3D viscoacoustic is not implemented; attenuation is silently ignored.
std::unique_ptr<mfem::BilinearFormIntegrator>
CreateAcousticStiffnessIntegrator3D(const AcousticMaterialBase3D& material,
                                    mfem::ParFiniteElementSpace& fes,
                                    real_t dt);

}  // namespace SEM

#endif  // SEM_STIFFNESS_INTEGRATOR_FACTORY_HPP
