/**
 * @file SrcRecvLoaders.hpp
 * @brief Source/Receiver loader functions from YamlConfig
 *
 * Separated from ConfigLoaders.hpp to break the circular dependency
 * between sem_material (which contains ConfigLoaders) and sem_srcrecv.
 */

#ifndef SEM_SRCRECV_LOADERS_HPP
#define SEM_SRCRECV_LOADERS_HPP

#include "config/YamlConfig.hpp"
#include "srcrecv/Source.hpp"
#include "srcrecv/Receiver.hpp"
#include <mfem.hpp>

namespace SEM {


using namespace mfem;

// =============================================================================
// Source Loading
// =============================================================================

/**
 * @brief Create source time function from SourceDef
 * @param src Source definition from YAML
 * @param nt Number of time steps
 * @param dt Time step
 * @param ncomp Number of components (2 for 2D force, 1 for scalar)
 * @return Source time function
 */
SourceTimeFunction CreateSTF(const SourceDef& src, int nt, real_t dt, int ncomp);

// =============================================================================
// Receiver Loading
// =============================================================================

/**
 * @brief Load receivers from YAML configuration
 * @param config YAML configuration
 * @param fes Finite element space
 * @param comm MPI communicator
 * @param domain Domain type (Solid/Fluid)
 * @param nt Number of time steps
 * @param dt Time step
 * @return Pointer to receiver array (caller owns memory)
 */
ReceiverArray* LoadReceivers(
    const YamlConfig& config,
    ParFiniteElementSpace& fes,
    MPI_Comm* comm,
    DomainType domain,
    int nt,
    real_t dt);

}  // namespace SEM

#endif  // SEM_SRCRECV_LOADERS_HPP
