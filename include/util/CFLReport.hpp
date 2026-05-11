/**
 * @file CFLReport.hpp
 * @brief Per-element CFL check + violation report, shared by pure and
 *        fluid-solid coupled facades.
 *
 * Wraps the element-wise Courant check
 *
 *     dt_max(e) = cfl_factor · h_min_gll(e) / v_max(e)
 *
 * with MPI-collective worst-element reduction and a single stderr report
 * formatted the same way as the original `SimulationFacade::CheckCFL` did.
 * Extracted so `CoupledSimulationFacade` can run the same check on each of
 * its sub-meshes without duplicating ~80 lines of loop + reporting logic.
 */

#ifndef SEM_UTIL_CFL_REPORT_HPP
#define SEM_UTIL_CFL_REPORT_HPP

#include <mfem.hpp>

#include <iostream>
#include <limits>
#include <string>

#include "material/MaterialBase.hpp"
#include "util/ElementMetrics.hpp"

namespace SEM {

/**
 * @brief Run a per-element CFL check on a (sub)mesh and return whether it
 *        is satisfied. Emits a WARNING block to stderr on violation.
 *
 * @tparam Dim Spatial dimension (2 or 3)
 * @param mesh           The (sub)mesh to walk.
 * @param order          H1 polynomial order used on this FES.
 * @param material       Material that answers `GetElementMaxVelocity(e)`.
 * @param dt             Configured time step.
 * @param cfl_factor     Safety factor (e.g. 0.3).
 * @param comm           MPI communicator for the reduction.
 * @param label          Short human-readable tag used in the warning
 *                       block (`"fluid"` / `"solid"` / `""` for single-
 *                       physics). Empty → no label.
 * @param abort_on_violation If true, MFEM_ABORT on the first violation
 *                       found (pure-simulation legacy behaviour).
 * @return true iff dt ≤ min_e dt_max(e) across all ranks.
 */
template<int Dim>
inline bool CheckCFLOnSubmesh(mfem::ParMesh&             mesh,
                              int                        order,
                              const MaterialBase&        material,
                              mfem::real_t               dt,
                              mfem::real_t               cfl_factor,
                              MPI_Comm                   comm,
                              const std::string&         label,
                              bool                       abort_on_violation)
{
    using mfem::real_t;

    const int ne = mesh.GetNE();

    int    local_violation_count = 0;
    real_t worst_dt_max_local    = std::numeric_limits<real_t>::max();
    int    worst_element_local   = -1;
    real_t worst_dist_local      = 0.0;
    real_t worst_v_local         = 0.0;

    for (int e = 0; e < ne; ++e) {
        const real_t dist_min   = ComputeElementMinGLLDistance<Dim>(mesh, e, order);
        const real_t v_elem     = material.GetElementMaxVelocity(e);
        const real_t dt_max_elem = cfl_factor * dist_min / v_elem;
        if (dt > dt_max_elem) local_violation_count++;
        if (dt_max_elem < worst_dt_max_local) {
            worst_dt_max_local  = dt_max_elem;
            worst_element_local = e;
            worst_dist_local    = dist_min;
            worst_v_local       = v_elem;
        }
    }

    // Global reductions
    int global_violation_count = 0;
    MPI_Allreduce(&local_violation_count, &global_violation_count, 1,
                  MPI_INT, MPI_SUM, comm);

    struct { double dt_max; int rank; } local_worst, global_worst;
    local_worst.dt_max = (worst_dt_max_local < std::numeric_limits<real_t>::max())
                         ? static_cast<double>(worst_dt_max_local) : 1e30;
    int myrank = 0;
    MPI_Comm_rank(comm, &myrank);
    local_worst.rank = myrank;
    MPI_Allreduce(&local_worst, &global_worst, 1,
                  MPI_DOUBLE_INT, MPI_MINLOC, comm);

    const real_t dt_max_global = static_cast<real_t>(global_worst.dt_max);
    const bool   satisfied     = (dt <= dt_max_global);

    if (!satisfied) {
        const std::string tag = label.empty() ? "CFL" : ("CFL " + label);
        if (myrank == 0) {
            std::cerr << "[" << tag << "] WARNING: CFL condition VIOLATED!\n";
            std::cerr << "[" << tag << "]   dt = " << dt
                      << " > dt_max = " << dt_max_global << "\n";
            std::cerr << "[" << tag << "]   " << global_violation_count
                      << " elements violate CFL condition\n";
        }
        if (myrank == global_worst.rank && worst_element_local >= 0) {
            mfem::Vector centroid(Dim);
            mesh.GetElementCenter(worst_element_local, centroid);
            std::cerr << "[" << tag << "]   Worst element: local_id="
                      << worst_element_local
                      << ", dt_max=" << worst_dt_max_local
                      << ", min_gll_dist=" << worst_dist_local << "m"
                      << ", v_max=" << worst_v_local << "m/s"
                      << ", centroid=(" << centroid(0);
            if (Dim >= 2) std::cerr << ", " << centroid(1);
            if (Dim >= 3) std::cerr << ", " << centroid(2);
            std::cerr << ")\n";
        }
        if (abort_on_violation) {
            MFEM_ABORT("CFL condition violated (" << (label.empty() ? "sim" : label)
                       << "): dt = " << dt << " > dt_max = " << dt_max_global);
        }
    }
    return satisfied;
}

}  // namespace SEM

#endif  // SEM_UTIL_CFL_REPORT_HPP
