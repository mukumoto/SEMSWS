/**
 * @file Operator.hpp
 * @brief Umbrella header for modern wave operators
 *
 * This file provides a single include for all wave operator classes:
 * - WaveOperator: Base class for 2nd order time-dependent operators
 * - ElasticOperator2D/3D: Isotropic elastic wave operators
 * - AcousticOperator2D/3D: Acoustic wave operators
 *
 * Key features:
 * - All data stored in MFEM Vectors (no raw pointer arrays)
 * - Builder pattern for chaining setup methods
 * - Uses BilinearFormIntegrator-based SEMIntegrators (PA compatible)
 * - Clean separation between setup and time stepping
 *
 * Example usage:
 *   #include "Operator.hpp"
 *
 *   SEM::ElasticOperator2D op(fes, order, material);
 *   op.SetupMass()
 *     .SetupStiffness()
 *     .SetupSource(source);
 *
 *   SEM::NewmarkCentralDifference integrator;
 *   integrator.Init(op, 0.0, dt);
 *   integrator.Step(u, dudt, dudt2, t, dt);
 */

#ifndef SEM_OPERATOR_HPP
#define SEM_OPERATOR_HPP

// Base class and configuration structures
#include "operator/WaveOperator.hpp"

// Damping utilities
#include "operator/damping/CerjanABC.hpp"

// Elastic operators
#include "operator/elastic/ElasticityOperator2D.hpp"
#include "operator/elastic/ElasticityOperator3D.hpp"

// Acoustic operators
#include "operator/acoustic/AcousticOperator2D.hpp"
#include "operator/acoustic/AcousticOperator3D.hpp"

namespace SEM {

}  // namespace SEM

#endif  // SEM_OPERATOR_HPP
