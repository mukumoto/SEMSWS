#include "util/MemoryReport.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>

namespace SEM {

double MemoryReport::GetMemoryUsageMB() {
    std::ifstream status("/proc/self/status");
    if (!status.is_open()) {
        return -1.0;
    }

    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            std::istringstream iss(line.substr(6));
            double kb;
            if (iss >> kb) {
                return kb / 1024.0;  // Convert KB to MB
            }
        }
    }
    return -1.0;
}

void MemoryReport::Record(const std::string& label) {
    Entry entry;
    entry.label = label;
    entry.memory_mb = GetMemoryUsageMB();
    entries_.push_back(entry);
}

void MemoryReport::Clear() {
    entries_.clear();
}

void MemoryReport::PrintSummary(std::ostream& os, MPI_Comm comm) const {
    if (entries_.empty()) {
        return;
    }

    int rank;
    MPI_Comm_rank(comm, &rank);

    // Collect local memory values
    std::vector<double> local_mem(entries_.size());
    for (size_t i = 0; i < entries_.size(); ++i) {
        local_mem[i] = entries_[i].memory_mb;
    }

    // MPI reduce for min/max/sum
    std::vector<double> min_mem(entries_.size());
    std::vector<double> max_mem(entries_.size());
    std::vector<double> sum_mem(entries_.size());

    MPI_Reduce(local_mem.data(), min_mem.data(),
               static_cast<int>(entries_.size()), MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(local_mem.data(), max_mem.data(),
               static_cast<int>(entries_.size()), MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(local_mem.data(), sum_mem.data(),
               static_cast<int>(entries_.size()), MPI_DOUBLE, MPI_SUM, 0, comm);

    // Only rank 0 prints
    if (rank != 0) {
        return;
    }

    // Find max label width for formatting
    size_t max_label_width = 10;  // minimum width
    for (const auto& entry : entries_) {
        if (entry.label.size() > max_label_width) {
            max_label_width = entry.label.size();
        }
    }

    os << "\n";
    os << "========================================\n";
    os << "        Memory Usage (MB)\n";
    os << "========================================\n";
    os << std::left << std::setw(static_cast<int>(max_label_width + 2)) << "Checkpoint"
       << std::right << std::setw(10) << "min"
       << std::setw(10) << "max"
       << std::setw(12) << "total" << "\n";
    os << std::string(max_label_width + 34, '-') << "\n";

    os << std::fixed << std::setprecision(1);
    for (size_t i = 0; i < entries_.size(); ++i) {
        os << std::left << std::setw(static_cast<int>(max_label_width + 2)) << entries_[i].label
           << std::right << std::setw(10) << min_mem[i]
           << std::setw(10) << max_mem[i]
           << std::setw(12) << sum_mem[i] << "\n";
    }

    os << "========================================\n";
    os << std::defaultfloat;
}

} // namespace SEM
