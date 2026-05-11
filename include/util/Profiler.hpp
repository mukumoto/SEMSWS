/**
 * @file Profiler.hpp
 * @brief Lightweight profiling infrastructure for SEMSWS
 *
 * Provides per-region timing with GPU synchronization support.
 * Enabled via cmake -DSEM_ENABLE_PROFILING=ON
 * Zero overhead when disabled (compile-time macros).
 */

#ifndef SEM_PROFILER_HPP
#define SEM_PROFILER_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <mpi.h>

// MFEM GPU sync macros
#ifdef MFEM_USE_CUDA
#include "general/cuda.hpp"
#endif
#ifdef MFEM_USE_HIP
#include "general/hip.hpp"
#endif

namespace SEM {

#ifdef SEM_ENABLE_PROFILING

/**
 * @brief Timer entry for a single profiling region
 */
struct TimerEntry {
    std::string name;
    int call_count = 0;
    double total_time = 0.0;   // seconds
    double min_time = 1e30;
    double max_time = 0.0;
    int depth = 0;             // nesting depth when first recorded
    bool gpu_sync = false;     // whether this region uses GPU sync
};

/**
 * @brief Stack frame for tracking nested regions
 */
struct RegionFrame {
    std::string name;
    double start_time;
    bool gpu_sync;
};

/**
 * @brief Singleton profiler class
 *
 * Usage:
 *   PROFILE_REGION("RegionName");      // CPU region
 *   PROFILE_REGION_GPU("GPUKernel");   // GPU region (with sync)
 */
class Profiler {
public:
    /**
     * @brief Get singleton instance
     */
    static Profiler& Instance() {
        static Profiler instance;
        return instance;
    }

    /**
     * @brief Begin a profiling region
     * @param name Region name
     * @param gpu_sync If true, synchronize GPU before timing
     */
    void BeginRegion(const std::string& name, bool gpu_sync = false) {
        if (gpu_sync) {
            SyncDevice();
        }

        double start = GetTime();
        int depth = static_cast<int>(region_stack_.size());

        region_stack_.push_back({name, start, gpu_sync});

        // Initialize timer entry if new
        auto it = timers_.find(name);
        if (it == timers_.end()) {
            TimerEntry entry;
            entry.name = name;
            entry.depth = depth;
            entry.gpu_sync = gpu_sync;
            timers_[name] = entry;
        }
    }

    /**
     * @brief End a profiling region
     * @param name Region name (must match BeginRegion)
     */
    void EndRegion(const std::string& name) {
        if (region_stack_.empty()) {
            std::cerr << "[Profiler] Warning: EndRegion called with empty stack: " << name << "\n";
            return;
        }

        RegionFrame& frame = region_stack_.back();
        if (frame.name != name) {
            std::cerr << "[Profiler] Warning: Region mismatch. Expected: " << frame.name
                      << ", Got: " << name << "\n";
        }

        if (frame.gpu_sync) {
            SyncDevice();
        }

        double end = GetTime();
        double elapsed = end - frame.start_time;

        // Update timer entry
        auto it = timers_.find(name);
        if (it != timers_.end()) {
            it->second.call_count++;
            it->second.total_time += elapsed;
            it->second.min_time = std::min(it->second.min_time, elapsed);
            it->second.max_time = std::max(it->second.max_time, elapsed);
        }

        region_stack_.pop_back();
    }

    /**
     * @brief RAII scoped region
     */
    class ScopedRegion {
    public:
        ScopedRegion(const std::string& name, bool gpu_sync = false)
            : name_(name) {
            Profiler::Instance().BeginRegion(name, gpu_sync);
        }

        ~ScopedRegion() {
            Profiler::Instance().EndRegion(name_);
        }

    private:
        std::string name_;
    };

    /**
     * @brief Print summary to output stream (single rank)
     */
    void PrintSummary(std::ostream& os = std::cout) const {
        os << "\n===== Profiling Summary =====\n\n";

        // Separate setup and time loop regions
        std::vector<const TimerEntry*> setup_entries;
        std::vector<const TimerEntry*> loop_entries;

        for (const auto& pair : timers_) {
            if (pair.first.find("Setup:") == 0) {
                setup_entries.push_back(&pair.second);
            } else {
                loop_entries.push_back(&pair.second);
            }
        }

        // Sort setup entries alphabetically (flat list, no hierarchy)
        auto setup_sorter = [](const TimerEntry* a, const TimerEntry* b) {
            return a->name < b->name;
        };
        std::sort(setup_entries.begin(), setup_entries.end(), setup_sorter);

        // Sort time loop entries by depth then name
        auto loop_sorter = [](const TimerEntry* a, const TimerEntry* b) {
            if (a->depth != b->depth) return a->depth < b->depth;
            return a->name < b->name;
        };
        std::sort(loop_entries.begin(), loop_entries.end(), loop_sorter);

        // Print setup regions
        if (!setup_entries.empty()) {
            os << "[Setup (one-time)]\n";
            os << std::left << std::setw(35) << "Region"
               << std::right << std::setw(8) << "Calls"
               << std::setw(12) << "Total(s)"
               << std::setw(12) << "Avg(ms)"
               << "\n";
            os << std::string(67, '-') << "\n";

            for (const auto* entry : setup_entries) {
                // No indentation for setup entries (flat list)
                double avg_ms = (entry->call_count > 0) ?
                    (entry->total_time / entry->call_count) * 1000.0 : 0.0;

                os << std::left << std::setw(35) << entry->name
                   << std::right << std::setw(8) << entry->call_count
                   << std::setw(12) << std::fixed << std::setprecision(3) << entry->total_time
                   << std::setw(12) << std::fixed << std::setprecision(2) << avg_ms
                   << "\n";
            }
            os << "\n";
        }

        // Print time loop regions
        if (!loop_entries.empty()) {
            os << "[Time Loop (per-step)]\n";
            os << std::left << std::setw(35) << "Region"
               << std::right << std::setw(8) << "Calls"
               << std::setw(12) << "Total(s)"
               << std::setw(12) << "Avg(ms)"
               << std::setw(12) << "Min(ms)"
               << std::setw(12) << "Max(ms)"
               << "\n";
            os << std::string(91, '-') << "\n";

            for (const auto* entry : loop_entries) {
                std::string indent(entry->depth * 2, ' ');
                double avg_ms = (entry->call_count > 0) ?
                    (entry->total_time / entry->call_count) * 1000.0 : 0.0;
                double min_ms = entry->min_time * 1000.0;
                double max_ms = entry->max_time * 1000.0;

                // Handle uninitialized min
                if (entry->call_count == 0) {
                    min_ms = 0.0;
                }

                os << std::left << std::setw(35) << (indent + entry->name)
                   << std::right << std::setw(8) << entry->call_count
                   << std::setw(12) << std::fixed << std::setprecision(3) << entry->total_time
                   << std::setw(12) << std::fixed << std::setprecision(2) << avg_ms
                   << std::setw(12) << std::fixed << std::setprecision(2) << min_ms
                   << std::setw(12) << std::fixed << std::setprecision(2) << max_ms
                   << "\n";
            }
        }

        os << "\n";
    }

    /**
     * @brief Print MPI-aware summary (collects stats across all ranks)
     * @param comm MPI communicator
     */
    void PrintMPISummary(MPI_Comm comm) const {
        int rank, size;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        // For now, just print from rank 0
        // TODO: Add MPI reduction for min/max/avg across ranks
        if (rank == 0) {
            PrintSummary(std::cout);

            if (size > 1) {
                std::cout << "[Note: MPI stats (min/max/avg across " << size
                          << " ranks) not yet implemented]\n\n";
            }
        }
    }

    /**
     * @brief Reset all timers
     */
    void Reset() {
        timers_.clear();
        region_stack_.clear();
    }

private:
    Profiler() = default;
    ~Profiler() = default;
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    /**
     * @brief Synchronize GPU device
     */
    void SyncDevice() {
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP)
        MFEM_DEVICE_SYNC;
#endif
    }

    /**
     * @brief Get current wall time
     */
    double GetTime() const {
        return MPI_Wtime();
    }

    std::unordered_map<std::string, TimerEntry> timers_;
    std::vector<RegionFrame> region_stack_;
};

// =============================================================================
// Profiling Macros
// =============================================================================

/**
 * @brief Profile a CPU region (no GPU sync)
 * Usage: PROFILE_REGION("RegionName");
 */
#define PROFILE_REGION(name) \
    SEM::Profiler::ScopedRegion _sem_prof_##__LINE__(name, false)

/**
 * @brief Profile a GPU region (with device sync for accurate timing)
 * Usage: PROFILE_REGION_GPU("GPUKernelName");
 */
#define PROFILE_REGION_GPU(name) \
    SEM::Profiler::ScopedRegion _sem_prof_##__LINE__(name, true)

#else  // SEM_ENABLE_PROFILING not defined

// =============================================================================
// No-op macros when profiling is disabled (zero overhead)
// =============================================================================

#define PROFILE_REGION(name) ((void)0)
#define PROFILE_REGION_GPU(name) ((void)0)

// Dummy Profiler class for when profiling is disabled
class Profiler {
public:
    static Profiler& Instance() {
        static Profiler instance;
        return instance;
    }
    void PrintSummary(std::ostream& = std::cout) const {}
    void PrintMPISummary(MPI_Comm) const {}
    void Reset() {}
};

#endif  // SEM_ENABLE_PROFILING

}  // namespace SEM

#endif  // SEM_PROFILER_HPP
