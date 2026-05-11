/**
 * @file ForwardSimulation.cpp
 * @brief Implementation of ForwardSimulation class
 */

#include "simulation/ForwardSimulation.hpp"
#include "config/ConfigLoaders.hpp"
#include "srcrecv/HDF5SourceReceiverWriter.hpp"
#include "util/Profiler.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>

namespace SEM {

// =============================================================================
// Constructor
// =============================================================================

template<int Dim>
ForwardSimulation<Dim>::ForwardSimulation(MPI_Comm comm)
    : SimulationFacade<Dim>(comm) {}

// =============================================================================
// Fluent Configuration
// =============================================================================

template<int Dim>
ForwardSimulation<Dim>& ForwardSimulation<Dim>::SetWavefieldWriter(
    std::unique_ptr<WavefieldWriter> writer) {
    wavefield_writers_.clear();
    wavefield_writers_.push_back(std::move(writer));
    return *this;
}

template<int Dim>
ForwardSimulation<Dim>& ForwardSimulation<Dim>::AddWavefieldWriter(
    std::unique_ptr<WavefieldWriter> writer) {
    wavefield_writers_.push_back(std::move(writer));
    return *this;
}

template<int Dim>
ForwardSimulation<Dim>& ForwardSimulation<Dim>::SetWavefieldWriters(
    std::vector<std::unique_ptr<WavefieldWriter>> writers) {
    wavefield_writers_ = std::move(writers);
    return *this;
}

template<int Dim>
ForwardSimulation<Dim>& ForwardSimulation<Dim>::EnableProgressOutput(int interval) {
    progress_enabled_ = (interval > 0);
    progress_interval_ = interval;

    // Set up log file in output directory
    std::string outdir = this->OutputDir();
    if (!outdir.empty() && progress_enabled_) {
        log_file_ = outdir + "/log.txt";
        // Create/truncate log file (rank 0 only)
        if (this->IsRoot()) {
            std::ofstream ofs(log_file_, std::ios::trunc);
        }
    }

    return *this;
}


// =============================================================================
// Run
// =============================================================================

template<int Dim>
void ForwardSimulation<Dim>::Run() {
    // Initialize if not already done
    const SimulationState& state = this->State();
    if (!state.is_initialized) {
        this->Initialize();
    }

    // Initialize wavefield writers
    for (auto& writer : wavefield_writers_) {
        writer->Init(this->Mesh(), this->NumSteps(), this->Comm());
    }

    double wall_start = MPI_Wtime();

    // Time stepping loop
    while (this->Step()) {
        const SimulationState& current_state = this->State();
        int step = current_state.step;

        // Default progress output with min/max (all ranks participate in MPI collective)
        if (progress_enabled_ && progress_interval_ > 0 && step % progress_interval_ == 0) {
            // MPI collective: all ranks compute local min/max then reduce
            real_t u_min = this->Displacement().Min();
            real_t u_max = this->Displacement().Max();
            real_t u_min_global, u_max_global;
            MPI_Allreduce(&u_min, &u_min_global, 1, HYPRE_MPI_REAL, MPI_MIN, this->Comm());
            MPI_Allreduce(&u_max, &u_max_global, 1, HYPRE_MPI_REAL, MPI_MAX, this->Comm());

            if (this->IsRoot()) {
                double elapsed = MPI_Wtime() - wall_start;
                double rate = (elapsed > 0) ? step / elapsed : 0.0;

                std::ostringstream line;
                line << "  Step " << std::setw(6) << step
                     << "/" << std::setw(6) << this->NumSteps()
                     << " | t=" << std::scientific << std::setprecision(3)
                     << current_state.time << " s"
                     << " | u=[" << std::setprecision(2)
                     << u_min_global << ", " << u_max_global << "]"
                     << std::fixed << std::setprecision(1)
                     << " | " << rate << " steps/s";

                std::cout << line.str() << std::endl;

                // Append to log file
                if (!log_file_.empty()) {
                    std::ofstream ofs(log_file_, std::ios::app);
                    if (ofs.is_open()) {
                        ofs << line.str() << "\n";
                    }
                }
            }
        }

        // Wavefield output at specified intervals
        int wf_source_id = -1;
        if (this->sources_ && this->sources_->NumSources() > 0) {
            wf_source_id = this->sources_->GetSource(0)->GetId();
        }
        for (auto& writer : wavefield_writers_) {
            if (writer->ShouldWrite(step)) {
                writer->Write(step, current_state.time,
                              &this->Displacement(),
                              &this->Velocity(),
                              &this->Acceleration(),
                              wf_source_id);
            }
        }
    }

    // Final wavefield output
    const SimulationState& final_state = this->State();
    int wf_source_id = -1;
    if (this->sources_ && this->sources_->NumSources() > 0) {
        wf_source_id = this->sources_->GetSource(0)->GetId();
    }
    for (auto& writer : wavefield_writers_) {
        if (writer->ShouldWrite(final_state.step)) {
            writer->Write(final_state.step, final_state.time,
                          &this->Displacement(),
                          &this->Velocity(),
                          &this->Acceleration(),
                          wf_source_id);
        }
    }

    // Finalize wavefield writers
    for (auto& writer : wavefield_writers_) {
        writer->Finalize();
    }

    // Save receivers; also embed /shots/0000/sources/ into the HDF5 output
    // so the file is self-roundtrip — its source metadata can be re-read
    // via the HDF5 input path.
    if (this->receivers_ && this->sources_ && this->sources_->NumSources() > 0) {
        int source_id = this->sources_->GetSource(0)->GetId();

        // Build source descriptors from the live YAML/HDF5 config; STF
        // samples come from SourceTimeFunction::FromConfig (deterministic).
        std::vector<HDF5SourceWriteEntry> src_entries;
        if constexpr (Dim == 2) {
            auto cfg = LoadSourceConfig2D(*this->config_);
            src_entries = BuildSourceWriteEntries(
                cfg, this->NumSteps(), this->Dt());
        } else {
            auto cfg = LoadSourceConfig3D(*this->config_);
            src_entries = BuildSourceWriteEntries(
                cfg, this->NumSteps(), this->Dt());
        }
        this->receivers_->SetOutputSourceContext(std::move(src_entries));

        this->receivers_->Save(source_id, this->T0(),
                               &this->sources_->GetSource(0)->Position());
    }

    // Record memory usage after simulation completes
    this->memory_report_.Record("RunComplete");

#ifdef SEM_ENABLE_PROFILING
    // Print profiling summary
    SEM::Profiler::Instance().PrintMPISummary(this->Comm());
#endif
}

// =============================================================================
// RunSequential
// =============================================================================

template<int Dim>
void ForwardSimulation<Dim>::RunSequential() {
    MFEM_VERIFY(this->sources_, "Sources must be set up before RunSequential");

    const int num_sources = this->sources_->NumSources();

    for (int src_id = 0; src_id < num_sources; src_id++) {
        // Reset all state: u,v,a=0, memory variables=0, receivers=0
        this->Reset();

        // Activate only this source
        this->sources_->SetActiveSource(src_id);

        // Initialize for this source
        this->Initialize();

        // Get config ID for this source (used for filenames)
        int config_id = this->sources_->GetSource(src_id)->GetId();

        // Initialize wavefield writers
        for (auto& writer : wavefield_writers_) {
            writer->Init(this->Mesh(), this->NumSteps(), this->Comm());
        }

        double wall_start = MPI_Wtime();

        // Time stepping loop
        while (this->Step()) {
            const SimulationState& current_state = this->State();
            int step = current_state.step;

            // Progress output
            if (progress_enabled_ && progress_interval_ > 0 && step % progress_interval_ == 0) {
                real_t u_min = this->Displacement().Min();
                real_t u_max = this->Displacement().Max();
                real_t u_min_global, u_max_global;
                MPI_Allreduce(&u_min, &u_min_global, 1, HYPRE_MPI_REAL, MPI_MIN, this->Comm());
                MPI_Allreduce(&u_max, &u_max_global, 1, HYPRE_MPI_REAL, MPI_MAX, this->Comm());

                if (this->IsRoot()) {
                    double elapsed = MPI_Wtime() - wall_start;
                    double rate = (elapsed > 0) ? step / elapsed : 0.0;

                    std::ostringstream line;
                    line << "  Step " << std::setw(6) << step
                         << "/" << std::setw(6) << this->NumSteps()
                         << " | t=" << std::scientific << std::setprecision(3)
                         << current_state.time << " s"
                         << " | u=[" << std::setprecision(2)
                         << u_min_global << ", " << u_max_global << "]"
                         << std::fixed << std::setprecision(1)
                         << " | " << rate << " steps/s";

                    std::cout << line.str() << std::endl;

                    if (!log_file_.empty()) {
                        std::ofstream ofs(log_file_, std::ios::app);
                        if (ofs.is_open()) {
                            ofs << line.str() << "\n";
                        }
                    }
                }
            }

            // Wavefield output at specified intervals
            for (auto& writer : wavefield_writers_) {
                if (writer->ShouldWrite(step)) {
                    writer->Write(step, current_state.time,
                                  &this->Displacement(),
                                  &this->Velocity(),
                                  &this->Acceleration(),
                                  config_id);
                }
            }
        }

        // Final wavefield output
        const SimulationState& final_state = this->State();
        for (auto& writer : wavefield_writers_) {
            if (writer->ShouldWrite(final_state.step)) {
                writer->Write(final_state.step, final_state.time,
                              &this->Displacement(),
                              &this->Velocity(),
                              &this->Acceleration(),
                              config_id);
            }
        }

        // Finalize wavefield writers
        for (auto& writer : wavefield_writers_) {
            writer->Finalize();
        }

        // Save receivers for this source. Embed /shots/0000/sources/ for
        // the active source only so each per-shot file is self-roundtrip
        // with that single source firing alone.
        std::vector<HDF5SourceWriteEntry> all_entries;
        if constexpr (Dim == 2) {
            auto cfg = LoadSourceConfig2D(*this->config_);
            all_entries = BuildSourceWriteEntries(
                cfg, this->NumSteps(), this->Dt());
        } else {
            auto cfg = LoadSourceConfig3D(*this->config_);
            all_entries = BuildSourceWriteEntries(
                cfg, this->NumSteps(), this->Dt());
        }
        std::vector<HDF5SourceWriteEntry> active_entries;
        for (auto& e : all_entries) {
            if (e.id == config_id) active_entries.push_back(std::move(e));
        }
        this->receivers_->SetOutputSourceContext(std::move(active_entries));

        this->receivers_->Save(config_id, this->T0(),
                               &this->sources_->GetSource(src_id)->Position());
    }

    // Restore all sources active
    this->sources_->ClearActiveSource();
    this->memory_report_.Record("RunComplete");
}


// =============================================================================
// Explicit Template Instantiations
// =============================================================================

template class ForwardSimulation<2>;
template class ForwardSimulation<3>;

}  // namespace SEM
