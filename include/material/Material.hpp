/**
 * @file Material.hpp
 * @brief Material system umbrella header
 *
 * This file includes all material-related headers.
 *
 * Material Hierarchy:
 * - MaterialBase: Common base (GetType, HasAttenuation)
 * - ElasticMaterialBase2D/3D: Elastic interface (Kappa, Mu, Rho)
 *     Note: Lambda is isotropic-specific, not in base class
 * - AcousticMaterialBase2D/3D: Acoustic interface (Kappa, InvRho)
 * - IsotropicElasticMaterial/3D: Concrete isotropic elastic (has Lambda)
 * - IsotropicAcousticMaterial/3D: Concrete isotropic acoustic
 */

#ifndef SEM_MATERIAL_HPP
#define SEM_MATERIAL_HPP

// Core material types
#include "material/MaterialField.hpp"
#include "material/MaterialBase.hpp"
#include "material/ElasticMaterialBase.hpp"
#include "material/AcousticMaterialBase.hpp"

// Concrete implementations
#include "material/isotropic_elastic/IsotropicElasticMaterial.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticMaterial.hpp"

#endif  // SEM_MATERIAL_HPP
