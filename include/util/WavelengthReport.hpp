/**
 * @file WavelengthReport.hpp
 * @brief Per-element points-per-wavelength check + warning report, shared
 *        by pure and fluid-solid coupled facades.
 *
 * Wraps the element-wise resolution check
 *
 *     ppw(e) = (N+1) · v_min(e) / (f_max · h_max(e))
 *
 * with MPI-collective worst-element reduction and a single stderr report
 * formatted the same way as the original `SimulationFacade::Check­Wavelength­Sampling`
 * did. Extracted so `CoupledSimulationFacade` can run the same check on
 * each of its sub-meshes without duplicating the loop + reduction logic.
 */

#ifndef SEM_UTIL_WAVELENGTH_REPORT_HPP
#define SEM_UTIL_WAVELENGTH_REPORT_HPP

#include <mfem.hpp>

#include <iostream>
#include <limits>
#include <string>

#include "material/MaterialBase.hpp"

namespace SEM {

/**
 * @brief Check that every element satisfies ppw >= ppw_required.
 *
 * @tparam Dim Spatial dimension (2 or 3).
 * @param mesh              The (sub)mesh to walk.
 * @param order             H1 polynomial order on this FES (N, not N+1).
 * @param material          Material that answers `GetElementMinVelocity(e)`.
 * @param f_max             Maximum source frequency (Hz).
 * @param ppw_required      Required points per wavelength (typically 5 for N=4).
 * @param comm              MPI communicator for the reductions.
 * @param label             Short tag used in the warning block
 *                          (`"fluid"` / `"solid"` / `""` for single-physics).
 * @param warn_on_insufficient If true, emit stderr warning on violation.
 * @return true iff every element across all ranks satisfies the PPW target.
 */
template<int Dim>
inline bool CheckWavelengthSamplingOnSubmesh(mfem::ParMesh&      mesh,
                                             int                 order,
                                             const MaterialBase& material,
                                             mfem::real_t        f_max,
                                             mfem::real_t        ppw_required,
                                             MPI_Comm            comm,
                                             const std::string&  label,
                                             bool                warn_on_insufficient)
{
    using mfem::real_t;

    const int ne   = mesh.GetNE();
    const int ngll = order + 1;

    int    local_insufficient_count = 0;
    real_t worst_ppw_local          = std::numeric_limits<real_t>::max();
    int    worst_element_local      = -1;
    real_t worst_h_local            = 0.0;
    real_t worst_v_local            = 0.0;

    for (int e = 0; e < ne; ++e) {
        const real_t h          = mesh.GetElementSize(e, 2);   // type=2 → h_max
        const real_t v_elem     = material.GetElementMinVelocity(e);
        const real_t lambda_e   = v_elem / f_max;
        const real_t ppw_elem   = ngll * lambda_e / h;
        if (ppw_elem < ppw_required) local_insufficient_count++;
        if (ppw_elem < worst_ppw_local) {
            worst_ppw_local     = ppw_elem;
            worst_element_local = e;
            worst_h_local       = h;
            worst_v_local       = v_elem;
        }
    }

    int global_insufficient_count = 0;
    MPI_Allreduce(&local_insufficient_count, &global_insufficient_count, 1,
                  MPI_INT, MPI_SUM, comm);

    const bool sufficient = (global_insufficient_count == 0);

    if (!sufficient && warn_on_insufficient) {
        struct { double ppw; int rank; } local_worst, global_worst;
        local_worst.ppw = (worst_ppw_local < std::numeric_limits<real_t>::max())
                           ? static_cast<double>(worst_ppw_local) : 1e30;
        int myrank = 0;
        MPI_Comm_rank(comm, &myrank);
        local_worst.rank = myrank;
        MPI_Allreduce(&local_worst, &global_worst, 1,
                      MPI_DOUBLE_INT, MPI_MINLOC, comm);

        const std::string tag = label.empty() ? "Resolution"
                                              : ("Resolution " + label);
        if (myrank == 0) {
            std::cerr << "[" << tag << "] WARNING: Wavelength sampling insufficient!\n";
            std::cerr << "[" << tag << "]   Elements with PPW < " << ppw_required
                      << ": " << global_insufficient_count << "\n";
            std::cerr << "[" << tag << "]   Consider: refining mesh or lowering"
                         " source frequency.\n";
        }
        if (myrank == global_worst.rank && worst_element_local >= 0) {
            mfem::Vector centroid(Dim);
            mesh.GetElementCenter(worst_element_local, centroid);
            std::cerr << "[" << tag << "]   Worst element: local_id="
                      << worst_element_local
                      << ", PPW=" << worst_ppw_local
                      << ", h=" << worst_h_local << "m"
                      << ", v_min=" << worst_v_local << "m/s"
                      << ", centroid=(" << centroid(0);
            if (Dim >= 2) std::cerr << ", " << centroid(1);
            if (Dim >= 3) std::cerr << ", " << centroid(2);
            std::cerr << ")\n";
        }
    }

    return sufficient;
}

}  // namespace SEM

#endif  // SEM_UTIL_WAVELENGTH_REPORT_HPP
