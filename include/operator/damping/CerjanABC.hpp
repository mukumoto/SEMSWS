/**
 * @file CerjanABC.hpp
 * @brief Cerjan sponge absorbing boundary condition utilities
 *
 * Implements the multiplicative taper method from:
 *   Cerjan et al., 1985, A nonreflecting boundary condition for discrete
 *   acoustic and elastic wave equations, Geophysics, 50(4), 705-708.
 *
 * The taper profile is:
 *   taper = exp(-a² · d²)
 *   a = sqrt(-log(1 - abpc/100))
 *
 * where:
 *   - d = normalized distance from interior boundary [0, 1]
 *   - abpc = absorption percentage (alpha parameter in config)
 *   - taper = 1.0 at interior (no damping), small at domain boundary
 *
 * Applied as a multiplicative factor on all state vectors (u, v, a)
 * after each complete time step.
 */

#ifndef SEM_OPERATOR_DAMPING_CERJAN_ABC_HPP
#define SEM_OPERATOR_DAMPING_CERJAN_ABC_HPP

#include <mfem.hpp>
#include <vector>

namespace SEM {


using namespace mfem;

// Forward declaration
struct DampingConfig;

namespace damping {

/**
 * @brief Compute Cerjan taper for 2D domains
 *
 * Computes taper = exp(-a² · d²) at each GLL point.
 * Interior: taper = 1.0, boundary: taper ≈ 1 - abpc/100.
 *
 * @param fes Parallel finite element space
 * @param config Damping configuration (abc_lengths, alpha as abpc%, attrs)
 * @param ngll Number of GLL points per direction
 * @param taper Output vector: taper value at each GLL point
 */
void ComputeCerjanTaper2D(
    ParFiniteElementSpace& fes,
    const DampingConfig& config,
    int ngll,
    Vector& taper);

/**
 * @brief Compute Cerjan taper for 3D domains
 *
 * @param fes Parallel finite element space
 * @param config Damping configuration
 * @param ngll Number of GLL points per direction
 * @param taper Output vector: taper value at each GLL point
 */
void ComputeCerjanTaper3D(
    ParFiniteElementSpace& fes,
    const DampingConfig& config,
    int ngll,
    Vector& taper);

/**
 * @brief Print Cerjan taper statistics (min/max values)
 *
 * @param taper Taper vector
 * @param comm MPI communicator
 * @param label Label for output
 */
void PrintCerjanStats(
    const Vector& taper,
    MPI_Comm comm,
    const char* label = "Cerjan taper");

}  // namespace damping
}  // namespace SEM

#endif  // SEM_OPERATOR_DAMPING_CERJAN_ABC_HPP
