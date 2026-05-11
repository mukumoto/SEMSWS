/**
 * @file ProgressLogger.hpp
 * @brief One-line-per-step progress output to stdout + log.txt, shared
 *        by pure ForwardSimulation and CoupledSimulationFacade Run loops.
 *
 * Format (single line, identical across simulations):
 *
 *     "  Step   1500/  4000 | t=7.500e-01 s | u=[-1.84e-01, 1.81e-01] | 108 steps/s"
 *
 * For the coupled run, two loggers are constructed (one per submesh) with
 * a "fluid" / "solid" prefix so the single log.txt under the shared output
 * directory stays human-readable:
 *
 *     "  [fluid] Step   1500/  4000 | t=7.500e-01 s | u=[...] | 108 steps/s"
 *     "  [solid] Step   1500/  4000 | t=7.500e-01 s | u=[...] | 108 steps/s"
 *
 * The class is stateless except for the log file path and the wall-clock
 * start time, so it's cheap to construct one per domain.
 */

#ifndef SEM_UTIL_PROGRESS_LOGGER_HPP
#define SEM_UTIL_PROGRESS_LOGGER_HPP

#include <mfem.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace SEM {

class ProgressLogger {
public:
    ProgressLogger() = default;

    /**
     * @brief Configure the logger. `interval <= 0` disables emission.
     *
     * @param interval   Emit one line every `interval` steps (plus the
     *                   final step). Use 0 to disable.
     * @param log_path   If non-empty, append progress lines to this file
     *                   (rank 0 only; caller should create/truncate before
     *                   the time loop so appends start from a clean file).
     * @param label      Short tag shown inside brackets before each line
     *                   ("fluid", "solid", or empty for unlabeled output).
     * @param comm       MPI communicator used for the u_min/u_max reduce.
     * @param is_root    Whether this rank prints — usually MPI rank 0.
     */
    void Configure(int interval,
                   const std::string& log_path,
                   const std::string& label,
                   MPI_Comm comm,
                   bool is_root)
    {
        interval_ = interval;
        log_path_ = log_path;
        label_    = label;
        comm_     = comm;
        is_root_  = is_root;
        wall_start_ = MPI_Wtime();

        // Rank 0 truncates the log file so re-runs start fresh.
        if (is_root_ && !log_path_.empty()) {
            std::ofstream ofs(log_path_, std::ios::trunc);
        }
    }

    bool Enabled() const { return interval_ > 0; }

    /**
     * @brief Emit one progress line if `step % interval == 0`. Safe to
     *        call on every step; does nothing when disabled or off-cadence.
     *
     * All ranks must call this collectively because u_min/u_max is
     * MPI_Allreduce'd across ranks.
     *
     * @param step   Current step index (0-based).
     * @param nt     Total number of steps.
     * @param t      Simulation time at `step`.
     * @param u_min_local / u_max_local  Per-rank min/max of the state
     *                                   component we want to display
     *                                   (typically displacement / potential).
     */
    void Tick(int step, int nt, mfem::real_t t,
              mfem::real_t u_min_local, mfem::real_t u_max_local)
    {
        if (interval_ <= 0) return;
        if (step % interval_ != 0) return;

        mfem::real_t u_min_global = 0.0;
        mfem::real_t u_max_global = 0.0;
        MPI_Allreduce(&u_min_local, &u_min_global, 1,
                      HYPRE_MPI_REAL, MPI_MIN, comm_);
        MPI_Allreduce(&u_max_local, &u_max_global, 1,
                      HYPRE_MPI_REAL, MPI_MAX, comm_);

        if (!is_root_) return;

        const double elapsed = MPI_Wtime() - wall_start_;
        const double rate    = (elapsed > 0.0 && step > 0) ? step / elapsed : 0.0;

        std::ostringstream line;
        line << "  ";
        if (!label_.empty()) line << "[" << label_ << "] ";
        line << "Step " << std::setw(6) << step
             << "/" << std::setw(6) << nt
             << " | t=" << std::scientific << std::setprecision(3) << t << " s"
             << " | u=[" << std::setprecision(2)
             << u_min_global << ", " << u_max_global << "]"
             << std::fixed << std::setprecision(1)
             << " | " << rate << " steps/s";

        std::cout << line.str() << std::endl;
        if (!log_path_.empty()) {
            std::ofstream ofs(log_path_, std::ios::app);
            if (ofs.is_open()) ofs << line.str() << "\n";
        }
    }

private:
    int         interval_ = 0;
    std::string log_path_;
    std::string label_;
    MPI_Comm    comm_     = MPI_COMM_WORLD;
    bool        is_root_  = false;
    double      wall_start_ = 0.0;
};

}  // namespace SEM

#endif  // SEM_UTIL_PROGRESS_LOGGER_HPP
