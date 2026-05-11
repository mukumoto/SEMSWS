/**
 * @file OperatorFactory.hpp
 * @brief Factory pattern for creating fully configured wave operators
 *
 * This file centralizes all operator creation logic:
 * - CreateWaveOperator2D: Creates and configures 2D operators
 * - CreateWaveOperator3D: Creates and configures 3D operators
 *
 * Supported material types:
 * - Isotropic (elastic)
 * - Acoustic
 * - Viscoelastic (elastic with material.HasAttenuation() == true)
 *
 * The operator type is automatically determined from MaterialType.
 * Viscoelasticity is enabled when the material has Q factors set.
 *
 * Example usage:
 *   OperatorConfig config;
 *   config.dt = 0.001;
 *   config.damping = DampingConfig(lengths, alphas, attrs);
 *
 *   auto op = CreateWaveOperator2D(fes, order, material, config);
 */

#ifndef SEM_OPERATOR_FACTORY_HPP
#define SEM_OPERATOR_FACTORY_HPP

#include "Operator.hpp"
#include "material/Material.hpp"
#include "material/ElasticMaterialBase.hpp"
#include "material/AcousticMaterialBase.hpp"
#include "common/Types.hpp"
#include <memory>

namespace SEM {

/**
 * @struct OperatorConfig
 * @brief Configuration for operator creation
 *
 * Contains all optional configuration for setting up a wave operator.
 * Default-constructed config creates a basic operator without damping.
 *
 * Note: Viscoelasticity is now controlled via material.HasAttenuation()
 * and Q values are read directly from the material.
 */
struct OperatorConfig {
    real_t dt = 0.0;                    ///< Time step (required for damping and viscoelastic)
    DampingConfig damping;              ///< Sponge ABC configuration
    Array<int> dirichlet_tdof;          ///< Dirichlet true DOFs (for RHS zeroing)
    Array<int> dirichlet_ldof;          ///< Dirichlet local DOFs (for state vector enforcement)

    OperatorConfig() = default;

    /// Check if damping is configured
    bool HasDamping() const {
        return !damping.IsEmpty() && dt > 0.0;
    }
};


// =============================================================================
// 2D Elastic Operators
// =============================================================================

/**
 * @brief Create and fully configure a 2D elastic wave operator
 *
 * @param fes Finite element space (vector H1, dim=2)
 * @param order Polynomial order
 * @param material Elastic material (any ElasticMaterialBase2D subclass)
 * @param config Optional configuration for damping
 * @return Fully configured elastic wave operator
 */
std::unique_ptr<WaveOperator> CreateElasticOperator2D(
    ParFiniteElementSpace& fes,
    int order,
    const ElasticMaterialBase2D& material,
    const OperatorConfig& config = OperatorConfig());

/**
 * @brief Create and fully configure a 2D acoustic wave operator
 *
 * @param fes Finite element space (scalar H1)
 * @param order Polynomial order
 * @param material Acoustic material (any AcousticMaterialBase2D subclass)
 * @param config Optional configuration for damping
 * @return Fully configured acoustic wave operator
 */
std::unique_ptr<WaveOperator> CreateAcousticOperator2D(
    ParFiniteElementSpace& fes,
    int order,
    const AcousticMaterialBase2D& material,
    const OperatorConfig& config = OperatorConfig());

// =============================================================================
// 3D Elastic Operators
// =============================================================================

/**
 * @brief Create and fully configure a 3D elastic wave operator
 *
 * @param fes Finite element space (vector H1, dim=3)
 * @param order Polynomial order
 * @param material Elastic material (any ElasticMaterialBase3D subclass)
 * @param config Optional configuration for damping
 * @return Fully configured elastic wave operator
 */
std::unique_ptr<WaveOperator> CreateElasticOperator3D(
    ParFiniteElementSpace& fes,
    int order,
    const ElasticMaterialBase3D& material,
    const OperatorConfig& config = OperatorConfig());

/**
 * @brief Create and fully configure a 3D acoustic wave operator
 *
 * @param fes Finite element space (scalar H1)
 * @param order Polynomial order
 * @param material Acoustic material (any AcousticMaterialBase3D subclass)
 * @param config Optional configuration for damping
 * @return Fully configured acoustic wave operator
 */
std::unique_ptr<WaveOperator> CreateAcousticOperator3D(
    ParFiniteElementSpace& fes,
    int order,
    const AcousticMaterialBase3D& material,
    const OperatorConfig& config = OperatorConfig());

}  // namespace SEM

#endif  // SEM_OPERATOR_FACTORY_HPP
