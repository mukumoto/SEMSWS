/**
 * @file SrcRecvLoaders.cpp
 * @brief Source/Receiver loader implementations
 *
 * Separated from ConfigLoaders.cpp to break circular dependency.
 */

#include "srcrecv/SrcRecvLoaders.hpp"
#include "common/Types.hpp"
#include <cmath>
#include <iostream>

namespace SEM {

// =============================================================================
// Dimension Validation
// =============================================================================

static void ValidateReceiverDimensions(const std::vector<ReceiverDef>& defs, int dim, int rank) {
    if (rank != 0) return;

    if (dim == 2) {
        for (size_t i = 0; i < defs.size(); i++) {
            const auto& def = defs[i];
            if (std::abs(def.location[2]) > 1e-10) {
                std::cerr << "WARNING: Receiver " << i << " (" << def.name
                          << ") has z-coordinate " << def.location[2]
                          << " but simulation is 2D. Z-coordinate will be ignored.\n";
            }
        }
    }
}

// =============================================================================
// Source Time Function
// =============================================================================

SourceTimeFunction CreateSTF(const SourceDef& src, int nt, real_t dt, int ncomp) {
    (void)ncomp;
    if (src.wavelet_type == "ricker") {
        SourceTimeFunction stf = SourceTimeFunction::Ricker(
            src.frequency, src.delay, dt, nt, src.amplitude);
        return stf;

    } else if (src.wavelet_type == "gaussian") {
        SourceTimeFunction stf = SourceTimeFunction::Gaussian(
            src.frequency, src.delay, dt, nt, src.amplitude);
        return stf;

    } else {
        return SourceTimeFunction::Ricker(src.frequency, src.delay, dt, nt, src.amplitude);
    }
}

// =============================================================================
// Receiver Loading
// =============================================================================

ReceiverArray* LoadReceivers(
    const YamlConfig& config,
    ParFiniteElementSpace& fes,
    MPI_Comm* comm,
    DomainType domain,
    int nt,
    real_t dt) {

    int dim = config.GetDimension();
    ReceiverArray* receivers = new ReceiverArray(fes, comm, domain);

    // External file and line expansion are handled by GetAllReceivers()
    // (YamlConfig::ParseReceivers loads from receivers.file if specified)

    std::vector<ReceiverDef> rec_list = config.GetAllReceivers();

    // Validate receiver dimensions
    int rank;
    MPI_Comm_rank(*comm, &rank);
    ValidateReceiverDimensions(rec_list, dim, rank);

    for (const auto& rec_def : rec_list) {
        Vector position(dim);
        for (int d = 0; d < dim; d++) {
            position(d) = rec_def.location[d];
        }

        // Filter receiver types by domain compatibility. Pressure is
        // fluid-only; displacement/velocity/acceleration/strain are
        // solid-only. Without this filter a coupled simulation would
        // try to locate a solid-domain station in the fluid submesh
        // (or vice versa) and abort with "Receiver not found in mesh".
        // For single-physics runs every receiver's types already match
        // the simulation's domain, so the filter is a no-op and does
        // not change behavior.
        bool any_added = false;
        for (const auto& type_str : rec_def.types) {
            ReceiverType rec_type = StringToReceiverType(type_str);
            const DomainType type_domain =
                (rec_type == ReceiverType::Pressure) ? DomainType::Fluid
                                                     : DomainType::Solid;
            if (type_domain != domain) continue;
            receivers->AddReceiver(rec_def.name, position, rec_type, rec_def.weight);
            any_added = true;
        }
        (void)any_added;  // silent skip is fine — station is on the other side
    }

    receivers->Setup(nt, dt);
    return receivers;
}

}  // namespace SEM
