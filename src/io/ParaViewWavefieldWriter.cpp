/**
 * @file ParaViewWavefieldWriter.cpp
 * @brief Implementation of ParaViewWavefieldWriter
 */

#include "io/ParaViewWavefieldWriter.hpp"

#include <sstream>
#include <iomanip>

#ifndef __linux__
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#endif

namespace SEM {

namespace {

bool CreateDirectory(const std::string& path) {
#ifdef __linux__
    if (mkdir(path.c_str(), 0755) == 0 || errno == EEXIST) {
#else
    if (_mkdir(path.c_str()) == 0 || errno == EEXIST) {
#endif
        return true;
    }
    return false;
}

}  // anonymous namespace

// =============================================================================
// Constructor
// =============================================================================

ParaViewWavefieldWriter::ParaViewWavefieldWriter(
    const std::string& output_dir, int interval, const Options& opts)
    : output_dir_(output_dir), interval_(interval), opts_(opts) {}

// =============================================================================
// Init
// =============================================================================

void ParaViewWavefieldWriter::Init(ParMesh& mesh, int total_steps, MPI_Comm comm) {
    MPI_Comm_rank(comm, &rank_);

    // Create base output directory (MFEM DataCollection creates subdirs)
    if (rank_ == 0) {
        CreateDirectory(output_dir_);
    }
    MPI_Barrier(comm);

    // Reset state for re-initialization (sequential source mode)
    initialized_ = false;
    pv_dc_.reset();
    u_copy_.reset();
    v_copy_.reset();
    a_copy_.reset();
}

// =============================================================================
// Deferred initialization (needs FESpace from actual GridFunction)
// =============================================================================

void ParaViewWavefieldWriter::InitCollection(const ParGridFunction* u) {
    if (initialized_ || !u) return;

    ParFiniteElementSpace* fes = u->ParFESpace();
    ParMesh* mesh = fes->GetParMesh();

    pv_dc_ = std::make_unique<ParaViewDataCollection>("paraview", mesh);
    pv_dc_->SetPrefixPath(output_dir_ + "/wavefield");
    pv_dc_->SetLevelsOfDetail(opts_.refinement);
    pv_dc_->SetHighOrderOutput(true);
    pv_dc_->SetDataFormat(ParseVTKFormat(opts_.data_format));
    // Only set compression if zlib is available (MFEM_USE_ZLIB)
#ifdef MFEM_USE_ZLIB
    if (opts_.compression >= 0) {
        pv_dc_->SetCompressionLevel(opts_.compression);
    }
#endif

    // Create owned copies and register fields
    if (opts_.write_displacement) {
        u_copy_ = std::make_unique<ParGridFunction>(fes);
        *u_copy_ = 0.0;
        pv_dc_->RegisterField("displacement", u_copy_.get());
    }
    if (opts_.write_pressure) {
        // For acoustic: u is scalar pressure. Reuse u_copy_ with "pressure" name.
        // If displacement is also registered (shouldn't happen), pressure gets its own copy.
        if (!u_copy_) {
            u_copy_ = std::make_unique<ParGridFunction>(fes);
            *u_copy_ = 0.0;
        }
        pv_dc_->RegisterField("pressure", u_copy_.get());
    }
    if (opts_.write_velocity) {
        v_copy_ = std::make_unique<ParGridFunction>(fes);
        *v_copy_ = 0.0;
        pv_dc_->RegisterField("velocity", v_copy_.get());
    }
    if (opts_.write_acceleration) {
        a_copy_ = std::make_unique<ParGridFunction>(fes);
        *a_copy_ = 0.0;
        pv_dc_->RegisterField("acceleration", a_copy_.get());
    }

    initialized_ = true;
}

// =============================================================================
// ShouldWrite
// =============================================================================

bool ParaViewWavefieldWriter::ShouldWrite(int step) const {
    return (step % interval_ == 0);
}

// =============================================================================
// Write
// =============================================================================

void ParaViewWavefieldWriter::Write(int step, real_t time,
                                     const ParGridFunction* u,
                                     const ParGridFunction* v,
                                     const ParGridFunction* a,
                                     int source_id) {
    // Deferred initialization on first write
    if (!initialized_) {
        InitCollection(u);
    }
    if (!pv_dc_) return;

    // Copy L-vector data into owned GridFunctions.
    // We use operator=(const Vector&) which copies the underlying data
    // without touching the T-vector (true DOF vector).
    // This avoids the need for SetTrueVector()/GetTrueVector() which
    // the simulation loop does not call.
    if (u_copy_ && u) {
        u_copy_->GridFunction::operator=(*u);
    }
    if (v_copy_ && v) {
        v_copy_->GridFunction::operator=(*v);
    }
    if (a_copy_ && a) {
        a_copy_->GridFunction::operator=(*a);
    }

    pv_dc_->SetCycle(step);
    pv_dc_->SetTime(time);
    pv_dc_->Save();
}

// =============================================================================
// Finalize
// =============================================================================

void ParaViewWavefieldWriter::Finalize() {
    // ParaViewDataCollection handles cleanup on destruction
    // Reset for potential re-use in sequential source mode
    initialized_ = false;
    pv_dc_.reset();
    u_copy_.reset();
    v_copy_.reset();
    a_copy_.reset();
}

// =============================================================================
// Helper
// =============================================================================

VTKFormat ParaViewWavefieldWriter::ParseVTKFormat(const std::string& fmt) {
    if (fmt == "ascii") return VTKFormat::ASCII;
    if (fmt == "binary32") return VTKFormat::BINARY32;
    return VTKFormat::BINARY;
}

}  // namespace SEM
