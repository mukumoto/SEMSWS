/**
 * @file WavefieldWriter.cpp
 * @brief Implementation of GLVisWavefieldWriter
 */

#include "io/WavefieldWriter.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cerrno>

#ifndef __linux__
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#endif

namespace SEM {

namespace {

/// Create directory (cross-platform, local to this file)
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
// GLVisWavefieldWriter Implementation
// =============================================================================

GLVisWavefieldWriter::GLVisWavefieldWriter(const std::string& output_dir, int interval)
    : output_dir_(output_dir), interval_(interval) {}

void GLVisWavefieldWriter::Init(ParMesh& mesh, int total_steps, MPI_Comm comm) {
    mesh_ = &mesh;
    MPI_Comm_rank(comm, &rank_);

    // Create output directories
    std::string glvis_dir = output_dir_ + "/wavefield/glvis";
    if (rank_ == 0) {
        CreateDirectory(output_dir_);
        CreateDirectory(output_dir_ + "/wavefield");
        CreateDirectory(glvis_dir);
    }
    MPI_Barrier(comm);
}

bool GLVisWavefieldWriter::ShouldWrite(int step) const {
    return (step % interval_ == 0);
}

void GLVisWavefieldWriter::Write(int step, real_t time,
                                  const ParGridFunction* u,
                                  const ParGridFunction* v,
                                  const ParGridFunction* a,
                                  int source_id) {
    std::string glvis_dir = output_dir_ + "/wavefield/glvis";

    // Build source ID prefix: "NNNN_" if source_id >= 0, empty otherwise
    std::string src_prefix;
    if (source_id >= 0) {
        std::ostringstream sid;
        sid << std::setfill('0') << std::setw(4) << source_id << "_";
        src_prefix = sid.str();
    }

    // Save mesh once on first write (SaveAsOne handles rank 0 ofstream internally)
    if (!mesh_saved_ && mesh_) {
        std::string mesh_file = glvis_dir + "/mesh.mesh";
        mesh_->SaveAsOne(mesh_file);
        mesh_saved_ = true;
    }

    // Save displacement field (primary output)
    if (u) {
        std::ostringstream oss;
        oss << glvis_dir << "/u_" << src_prefix
            << std::setfill('0') << std::setw(6) << step << ".gf";
        std::ofstream u_ofs(oss.str());
        if (u_ofs.good()) {
            u->SaveAsOne(u_ofs);
        }
    }
}

void GLVisWavefieldWriter::Finalize() {
    // Nothing to finalize for GLVis output
}

}  // namespace SEM
