/**
 * @file FieldVizWriter.cpp
 * @brief Implementation of FieldVizWriter
 *
 * Logic lifted from MaterialWriter so the runtime-material path and the
 * offline `sem_viz` tool share a single implementation. The only change
 * from the old MaterialWriter is the extra `subdir` argument that lets
 * callers pick the category folder ("material", "kernels", "", …).
 */

#include "io/FieldVizWriter.hpp"
#include "io/GMTWavefieldWriter.hpp"
#include "util/PointFinder.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cerrno>
#include <cmath>

#ifndef __linux__
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#endif

namespace SEM {
namespace FieldVizWriter {

namespace {

bool CreateDir(const std::string& path) {
#ifdef __linux__
    return (mkdir(path.c_str(), 0755) == 0 || errno == EEXIST);
#else
    return (_mkdir(path.c_str()) == 0 || errno == EEXIST);
#endif
}

std::string JoinDir(const std::string& base, const std::string& sub) {
    if (sub.empty()) return base;
    if (base.empty()) return sub;
    return base + "/" + sub;
}

}  // anonymous namespace

// =============================================================================
// WriteParaView
// =============================================================================

void WriteParaView(const std::string& output_dir,
                   const std::string& subdir,
                   const FieldList& fields,
                   const OutputFormatConfig& fmt,
                   ParMesh* mesh) {
    const std::string prefix = JoinDir(output_dir, subdir);

    ParaViewDataCollection pv_dc("paraview", mesh);
    pv_dc.SetPrefixPath(prefix);
    pv_dc.SetLevelsOfDetail(fmt.refinement);
    pv_dc.SetHighOrderOutput(true);

    VTKFormat vtk_fmt = VTKFormat::BINARY32;
    if (fmt.data_format == "ascii") vtk_fmt = VTKFormat::ASCII;
    else if (fmt.data_format == "binary") vtk_fmt = VTKFormat::BINARY;
    pv_dc.SetDataFormat(vtk_fmt);

#ifdef MFEM_USE_ZLIB
    if (fmt.compression >= 0) {
        pv_dc.SetCompressionLevel(fmt.compression);
    }
#endif

    for (const auto& [name, gf] : fields) {
        pv_dc.RegisterField(name, gf);
    }

    pv_dc.SetCycle(0);
    pv_dc.SetTime(0.0);
    pv_dc.Save();
}

// =============================================================================
// WriteGLVis
// =============================================================================

void WriteGLVis(const std::string& output_dir,
                const std::string& subdir,
                const FieldList& fields) {
    if (fields.empty()) return;

    const std::string cat_dir = JoinDir(output_dir, subdir);
    const std::string glvis_dir = JoinDir(cat_dir, "glvis");

    int rank = 0;
    MPI_Comm_rank(fields[0].second->ParFESpace()->GetComm(), &rank);
    if (rank == 0) {
        CreateDir(output_dir);
        if (!subdir.empty()) CreateDir(cat_dir);
        CreateDir(glvis_dir);
    }
    MPI_Barrier(fields[0].second->ParFESpace()->GetComm());

    for (const auto& [name, gf] : fields) {
        const std::string filename = glvis_dir + "/" + name + ".gf";
        std::ofstream ofs(filename);
        if (ofs.good()) {
            gf->SaveAsOne(ofs);
        }
    }

    ParMesh* mesh = fields[0].second->ParFESpace()->GetParMesh();
    mesh->SaveAsOne(glvis_dir + "/mesh.mesh");
}

// =============================================================================
// WriteGMT
// =============================================================================

void WriteGMT(const std::string& output_dir,
              const std::string& subdir,
              const FieldList& fields,
              const OutputFormatConfig& fmt,
              ParMesh& mesh,
              MPI_Comm comm) {
    int rank, dim;
    MPI_Comm_rank(comm, &rank);
    dim = mesh.Dimension();

    const std::string cat_dir = JoinDir(output_dir, subdir);
    const std::string gmt_dir = JoinDir(cat_dir, "gmt");

    if (rank == 0) {
        CreateDir(output_dir);
        if (!subdir.empty()) CreateDir(cat_dir);
        CreateDir(gmt_dir);
    }
    MPI_Barrier(comm);

    std::vector<GMTGrid> grids;

    Vector bb_min, bb_max;
    mesh.GetBoundingBox(bb_min, bb_max);

    if (dim == 2) {
        GMTGrid grid;
        grid.name = "";
        grid.nx = fmt.resolution[0];
        grid.ny = fmt.resolution[1];
        grid.x_min = bb_min(0); grid.x_max = bb_max(0);
        grid.y_min = bb_min(1); grid.y_max = bb_max(1);
        grid.npts = grid.nx * grid.ny;

        grid.grid_x.resize(grid.nx);
        grid.grid_y.resize(grid.ny);
        for (int i = 0; i < grid.nx; ++i)
            grid.grid_x[i] = grid.x_min + i * (grid.x_max - grid.x_min) / (grid.nx - 1);
        for (int j = 0; j < grid.ny; ++j)
            grid.grid_y[j] = grid.y_min + j * (grid.y_max - grid.y_min) / (grid.ny - 1);

        grid.interp_points.SetSize(dim * grid.npts);
        int idx = 0;
        for (int j = 0; j < grid.ny; ++j)
            for (int i = 0; i < grid.nx; ++i) {
                grid.interp_points(idx++) = grid.grid_x[i];
                grid.interp_points(idx++) = grid.grid_y[j];
            }

        grid.finder = std::make_unique<PointFinder>(comm);
        grid.finder->Setup(mesh);
        grid.finder->FindPoints(grid.interp_points);
        grids.push_back(std::move(grid));

    } else if (dim == 3) {
        GMTCrossSections cs = fmt.cross_sections;
        if (cs.Empty()) {
            const real_t xmid = 0.5 * (bb_min(0) + bb_max(0));
            const real_t ymid = 0.5 * (bb_min(1) + bb_max(1));
            const real_t zmid = 0.5 * (bb_min(2) + bb_max(2));
            cs.yz.push_back(xmid);
            cs.xz.push_back(ymid);
            cs.xy.push_back(zmid);
            if (rank == 0) {
                MFEM_WARNING("FieldVizWriter::WriteGMT: 3D input without "
                             "cross_sections; defaulting to mid-plane "
                             "slices at x=" << xmid << ", y=" << ymid
                             << ", z=" << zmid << ".");
            }
        }

        auto setup_slice = [&](const std::string& name, int ax1, int ax2, int ax_fixed,
                               real_t fixed_val) {
            GMTGrid grid;
            grid.name = name;
            grid.nx = fmt.resolution[0];
            grid.ny = fmt.resolution[1];
            grid.npts = grid.nx * grid.ny;
            grid.x_min = bb_min(ax1); grid.x_max = bb_max(ax1);
            grid.y_min = bb_min(ax2); grid.y_max = bb_max(ax2);

            grid.grid_x.resize(grid.nx);
            grid.grid_y.resize(grid.ny);
            for (int i = 0; i < grid.nx; ++i)
                grid.grid_x[i] = grid.x_min + i * (grid.x_max - grid.x_min) / (grid.nx - 1);
            for (int j = 0; j < grid.ny; ++j)
                grid.grid_y[j] = grid.y_min + j * (grid.y_max - grid.y_min) / (grid.ny - 1);

            grid.interp_points.SetSize(3 * grid.npts);
            int idx = 0;
            for (int j = 0; j < grid.ny; ++j)
                for (int i = 0; i < grid.nx; ++i) {
                    Vector pt(3); pt = 0.0;
                    pt(ax1) = grid.grid_x[i];
                    pt(ax2) = grid.grid_y[j];
                    pt(ax_fixed) = fixed_val;
                    grid.interp_points(idx++) = pt(0);
                    grid.interp_points(idx++) = pt(1);
                    grid.interp_points(idx++) = pt(2);
                }

            if (rank == 0) CreateDir(gmt_dir + "/" + name);
            grid.finder = std::make_unique<PointFinder>(comm);
            grid.finder->Setup(mesh);
            grid.finder->FindPoints(grid.interp_points);
            grids.push_back(std::move(grid));
        };

        MPI_Barrier(comm);

        for (real_t x : cs.yz) {
            std::ostringstream n; n << "yz_x" << x;
            setup_slice(n.str(), 1, 2, 0, x);
        }
        for (real_t y : cs.xz) {
            std::ostringstream n; n << "xz_y" << y;
            setup_slice(n.str(), 0, 2, 1, y);
        }
        for (real_t z : cs.xy) {
            std::ostringstream n; n << "xy_z" << z;
            setup_slice(n.str(), 0, 1, 2, z);
        }
    }

    if (rank == 0) {
        for (const auto& grid : grids) {
            grid.WriteGridInfo(gmt_dir);
        }
    }

    for (const auto& grid : grids) {
        const auto& elem = grid.finder->GetElem();
        const auto& proc = grid.finder->GetProc();
        const auto& code = grid.finder->GetCode();
        const auto& ref_pos = grid.finder->GetReferencePosition();

        for (const auto& [field_name, gf] : fields) {
            std::vector<real_t> local_values(grid.npts, 0.0);
            std::vector<int> local_found(grid.npts, 0);

            for (int p = 0; p < grid.npts; ++p) {
                if (code[p] == 2) continue;
                if (static_cast<int>(proc[p]) != rank) continue;
                int e = static_cast<int>(elem[p]);
                if (e < 0) continue;

                Vector ref_pt(dim);
                for (int d = 0; d < dim; ++d)
                    ref_pt(d) = ref_pos(p * dim + d);

                IntegrationPoint ip;
                ip.Set(ref_pt.GetData(), dim);

                local_values[p] = gf->GetValue(e, ip);
                local_found[p] = 1;
            }

            std::vector<real_t> global_values(grid.npts, 0.0);
            std::vector<int> global_found(grid.npts, 0);
            MPI_Reduce(local_values.data(), global_values.data(), grid.npts,
                       HYPRE_MPI_REAL, MPI_SUM, 0, comm);
            MPI_Reduce(local_found.data(), global_found.data(), grid.npts,
                       MPI_INT, MPI_SUM, 0, comm);

            if (rank == 0) {
                std::string filename;
                if (grid.name.empty()) {
                    filename = gmt_dir + "/" + field_name + ".xyz";
                } else {
                    filename = gmt_dir + "/" + grid.name + "/" + field_name + ".xyz";
                }

                std::ofstream ofs(filename);
                if (!ofs.good()) continue;

                ofs << std::scientific << std::setprecision(8);
                for (int j = 0; j < grid.ny; ++j)
                    for (int i = 0; i < grid.nx; ++i) {
                        int p = j * grid.nx + i;
                        ofs << grid.grid_x[i] << " " << grid.grid_y[j] << " ";
                        if (global_found[p] > 0)
                            ofs << global_values[p];
                        else
                            ofs << "NaN";
                        ofs << "\n";
                    }
            }
        }
    }
}

}  // namespace FieldVizWriter
}  // namespace SEM
