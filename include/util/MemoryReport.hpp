#pragma once
/**
 * @file MemoryReport.hpp
 * @brief Simple memory usage reporting based on /proc/self/status (Linux)
 *
 * Records memory usage at various checkpoints during simulation setup and run,
 * then outputs a summary table with MPI-aggregated statistics.
 */

#include <string>
#include <vector>
#include <ostream>
#include <mpi.h>

namespace SEM {

// Reads VmRSS from /proc/self/status (Linux-only) at labeled checkpoints;
// PrintSummary() aggregates min/max/total across MPI ranks.
class MemoryReport {
public:
    /**
     * @brief Record current memory usage with a label
     * @param label Checkpoint name (e.g., "MeshLoaded", "SetupComplete")
     */
    void Record(const std::string& label);

    /**
     * @brief Clear all recorded entries
     */
    void Clear();

    /**
     * @brief Check if any entries have been recorded
     * @return true if at least one entry exists
     */
    bool HasEntries() const { return !entries_.empty(); }

    /**
     * @brief Get number of recorded entries
     * @return Number of checkpoints recorded
     */
    size_t NumEntries() const { return entries_.size(); }

    /**
     * @brief Print summary table with MPI-aggregated statistics
     *
     * Performs MPI_Reduce to compute min/max/sum across all ranks.
     * Only rank 0 outputs to the stream.
     *
     * @param os Output stream (only used on rank 0)
     * @param comm MPI communicator
     */
    void PrintSummary(std::ostream& os, MPI_Comm comm) const;

private:
    /**
     * @brief Get current memory usage in MB from /proc/self/status
     * @return Memory usage in MB, or -1.0 if unavailable
     */
    static double GetMemoryUsageMB();

    struct Entry {
        std::string label;
        double memory_mb;
    };

    std::vector<Entry> entries_;
};

} // namespace SEM
