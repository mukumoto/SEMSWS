/**
 * @file SEMIntegrators.hpp
 * @brief Umbrella header for all SEM integrators
 *
 * This header provides access to all SEM integrators through a single include.
 *
 * Available integrators:
 *
 * Isotropic Acoustic (scalar wave equation):
 * - SEMIsotropicAcousticIntegrator2D
 * - SEMIsotropicAcousticIntegrator3D
 *
 * Isotropic Elastic:
 * - SEMIsotropicElasticIntegrator2D
 * - SEMIsotropicElasticIntegrator3D
 *
 * Visco_IsotropicElastic (isotropic elastic with Generalized Zener attenuation):
 * - SEMVisco_IsotropicElasticIntegrator2D
 * - SEMVisco_IsotropicElasticIntegrator3D
 *
 * Mass (diagonal mass matrix):
 * - SEMMassIntegrator2D
 * - SEMMassIntegrator3D
 *
 * Core components (for building custom integrators):
 * - SEMGeometry2D, SEMGeometry3D
 * - SEMDofOrdering2D, SEMDofOrdering3D
 * - SEMIntegratorBase2D, SEMIntegratorBase3D
 *
 * Usage:
 * @code
 * #include "SEMIntegrators.hpp"
 *
 * // Create acoustic integrator
 * SEM::SEMIsotropicAcousticIntegrator2D acoustic_integ(inv_rho);
 * acoustic_integ.AssemblePA(fes);
 * acoustic_integ.AddMultPA(p, f);
 *
 * // Create elastic integrator
 * SEM::SEMIsotropicElasticIntegrator2D elastic_integ(lambda, mu);
 * elastic_integ.AssemblePA(fes);
 * elastic_integ.AddMultPA(u, f);
 *
 * // Create viscoelastic integrator
 * SEM::SEMVisco_IsotropicElasticIntegrator2D visco_integ(lambda, mu);
 * visco_integ.GetAttenuation().EnableAttenuation(Qkappa, Qmu, f0, n_sls, dt);
 * visco_integ.AssemblePA(fes);
 * visco_integ.AddMultPA(u, f);
 * @endcode
 */

#ifndef SEM_INTEGRATORS_HPP
#define SEM_INTEGRATORS_HPP

// Core components
#include "integ/core/SEMGeometry.hpp"
#include "integ/core/SEMDofOrdering.hpp"
#include "integ/core/SEMIntegratorBase.hpp"
#include "integ/core/SEMKernelDispatch.hpp"

// Attenuation model framework
#include "integ/attenuation/AttenuationModel.hpp"
#include "integ/attenuation/GeneralizedZener.hpp"

// Direct integrator classes (no templates)
#include "integ/SEMIsotropicAcousticIntegrator.hpp"
#include "integ/SEMIsotropicElasticIntegrator.hpp"
#include "integ/SEMVisco_IsotropicElasticIntegrator.hpp"

// Mass integrators
#include "integ/SEMMassIntegrator.hpp"

#endif  // SEM_INTEGRATORS_HPP
