/**
 * @file GMTWavefieldWriter.cpp
 * @brief Implementation of GMTWavefieldWriter
 */

#include "io/GMTWavefieldWriter.hpp"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <limits>

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
// GMTGrid::WriteGridInfo
// =============================================================================

void GMTGrid::WriteGridInfo(const std::string& dir) const {
    std::string filepath = dir + "/grid_info.txt";
    if (!name.empty()) {
        filepath = dir + "/" + name + "/grid_info.txt";
    }
    std::ofstream ofs(filepath);
    if (!ofs.good()) return;

    real_t dx = (nx > 1) ? (x_max - x_min) / (nx - 1) : 0.0;
    real_t dy = (ny > 1) ? (y_max - y_min) / (ny - 1) : 0.0;

    ofs << std::setprecision(15);
    ofs << "# GMT grid info (auto-generated)\n";
    ofs << "nx=" << nx << "\n";
    ofs << "ny=" << ny << "\n";
    ofs << "x_min=" << x_min << "\n";
    ofs << "x_max=" << x_max << "\n";
    ofs << "y_min=" << y_min << "\n";
    ofs << "y_max=" << y_max << "\n";
    ofs << "dx=" << dx << "\n";
    ofs << "dy=" << dy << "\n";
    ofs << "region=" << x_min << "/" << x_max << "/" << y_min << "/" << y_max << "\n";
    ofs << "spacing=" << dx << "/" << dy << "\n";
}

// =============================================================================
// Constructor
// =============================================================================

GMTWavefieldWriter::GMTWavefieldWriter(
    const std::string& output_dir, int interval, const Options& opts)
    : output_dir_(output_dir), interval_(interval), opts_(opts) {}

// =============================================================================
// Init
// =============================================================================

void GMTWavefieldWriter::Init(ParMesh& mesh, int total_steps, MPI_Comm comm) {
    comm_ = comm;
    MPI_Comm_rank(comm, &rank_);
    dim_ = mesh.Dimension();

    // Create output directories
    std::string gmt_dir = output_dir_ + "/wavefield/gmt";
    if (rank_ == 0) {
        CreateDirectory(output_dir_);
        CreateDirectory(output_dir_ + "/wavefield");
        CreateDirectory(gmt_dir);
    }
    MPI_Barrier(comm);

    // Clear previous grids (for sequential source re-init)
    grids_.clear();

    if (dim_ == 2) {
        Setup2DGrid(mesh);
    } else if (dim_ == 3) {
        Setup3DGrids(mesh);
    }

    // Write grid info header files (rank 0 only)
    if (rank_ == 0) {
        std::string gmt_dir = output_dir_ + "/wavefield/gmt";
        for (const auto& grid : grids_) {
            grid.WriteGridInfo(gmt_dir);
        }
    }
}

// =============================================================================
// Setup 2D Grid
// =============================================================================

void GMTWavefieldWriter::SetBoundingBoxOverride(const Vector& bb_min,
                                                 const Vector& bb_max)
{
    has_bbox_override_ = true;
    bbox_min_override_.SetSize(bb_min.Size());
    bbox_min_override_ = bb_min;
    bbox_max_override_.SetSize(bb_max.Size());
    bbox_max_override_ = bb_max;
}

void GMTWavefieldWriter::Setup2DGrid(ParMesh& mesh) {
    Vector bb_min, bb_max;
    if (has_bbox_override_) {
        // Coupled path: use the parent-mesh bbox so fluid & solid
        // writers share a common grid; PointFinder tags points outside
        // this submesh as "not found" → written as NaN.
        bb_min = bbox_min_override_;
        bb_max = bbox_max_override_;
    } else {
        mesh.GetBoundingBox(bb_min, bb_max);
    }

    GMTGrid grid;
    grid.name = "";  // No subdirectory for 2D
    grid.nx = opts_.resolution[0];
    grid.ny = opts_.resolution[1];
    grid.x_min = bb_min(0);
    grid.x_max = bb_max(0);
    grid.y_min = bb_min(1);
    grid.y_max = bb_max(1);
    grid.axis1 = "x";
    grid.axis2 = "y";
    grid.npts = grid.nx * grid.ny;

    // Build 1D coordinate arrays
    grid.grid_x.resize(grid.nx);
    grid.grid_y.resize(grid.ny);
    for (int i = 0; i < grid.nx; ++i) {
        grid.grid_x[i] = grid.x_min + i * (grid.x_max - grid.x_min) / (grid.nx - 1);
    }
    for (int j = 0; j < grid.ny; ++j) {
        grid.grid_y[j] = grid.y_min + j * (grid.y_max - grid.y_min) / (grid.ny - 1);
    }

    // Build interpolation points [dim * npts] in byVDIM ordering
    grid.interp_points.SetSize(dim_ * grid.npts);
    int idx = 0;
    for (int j = 0; j < grid.ny; ++j) {
        for (int i = 0; i < grid.nx; ++i) {
            grid.interp_points(idx++) = grid.grid_x[i];
            grid.interp_points(idx++) = grid.grid_y[j];
        }
    }

    // Setup PointFinder and find points
    grid.finder = std::make_unique<PointFinder>(comm_);
    grid.finder->Setup(mesh);
    grid.finder->FindPoints(grid.interp_points);

    grids_.push_back(std::move(grid));
}

// =============================================================================
// Setup 3D Grids (Cross-Sections)
// =============================================================================

void GMTWavefieldWriter::Setup3DGrids(ParMesh& mesh) {
    const auto& cs = opts_.cross_sections;

    if (cs.Empty()) {
        if (rank_ == 0) {
            std::cerr << "WARNING: GMT output for 3D requires cross_sections. "
                      << "No GMT output will be produced." << std::endl;
        }
        return;
    }

    Vector bb_min, bb_max;
    if (has_bbox_override_) {
        bb_min = bbox_min_override_;
        bb_max = bbox_max_override_;
    } else {
        mesh.GetBoundingBox(bb_min, bb_max);
    }

    // Create subdirectories and grids
    std::string gmt_dir = output_dir_ + "/wavefield/gmt";

    // yz-planes (fixed x)
    for (real_t x_pos : cs.yz) {
        std::ostringstream name;
        name << "yz_x" << x_pos;
        if (rank_ == 0) {
            CreateDirectory(gmt_dir + "/" + name.str());
        }
        SetupSliceGrid(name.str(), "yz", x_pos, bb_min, bb_max);
    }

    // xz-planes (fixed y)
    for (real_t y_pos : cs.xz) {
        std::ostringstream name;
        name << "xz_y" << y_pos;
        if (rank_ == 0) {
            CreateDirectory(gmt_dir + "/" + name.str());
        }
        SetupSliceGrid(name.str(), "xz", y_pos, bb_min, bb_max);
    }

    // xy-planes (fixed z)
    for (real_t z_pos : cs.xy) {
        std::ostringstream name;
        name << "xy_z" << z_pos;
        if (rank_ == 0) {
            CreateDirectory(gmt_dir + "/" + name.str());
        }
        SetupSliceGrid(name.str(), "xy", z_pos, bb_min, bb_max);
    }

    MPI_Barrier(comm_);

    // Setup PointFinder for each grid
    ParMesh* mesh_ptr = &mesh;
    for (auto& grid : grids_) {
        grid.finder = std::make_unique<PointFinder>(comm_);
        grid.finder->Setup(*mesh_ptr);
        grid.finder->FindPoints(grid.interp_points);
    }
}

void GMTWavefieldWriter::SetupSliceGrid(
    const std::string& name,
    const std::string& plane,
    real_t fixed_coord,
    const Vector& bb_min, const Vector& bb_max) {

    GMTGrid grid;
    grid.name = name;
    grid.nx = opts_.resolution[0];
    grid.ny = opts_.resolution[1];
    grid.npts = grid.nx * grid.ny;

    // Determine axes based on plane
    int ax1, ax2, ax_fixed;
    if (plane == "yz") {
        ax1 = 1; ax2 = 2; ax_fixed = 0;
        grid.axis1 = "y"; grid.axis2 = "z";
    } else if (plane == "xz") {
        ax1 = 0; ax2 = 2; ax_fixed = 1;
        grid.axis1 = "x"; grid.axis2 = "z";
    } else {  // "xy"
        ax1 = 0; ax2 = 1; ax_fixed = 2;
        grid.axis1 = "x"; grid.axis2 = "y";
    }

    grid.x_min = bb_min(ax1);
    grid.x_max = bb_max(ax1);
    grid.y_min = bb_min(ax2);
    grid.y_max = bb_max(ax2);

    // Build 1D coordinate arrays
    grid.grid_x.resize(grid.nx);
    grid.grid_y.resize(grid.ny);
    for (int i = 0; i < grid.nx; ++i) {
        grid.grid_x[i] = grid.x_min + i * (grid.x_max - grid.x_min) / (grid.nx - 1);
    }
    for (int j = 0; j < grid.ny; ++j) {
        grid.grid_y[j] = grid.y_min + j * (grid.y_max - grid.y_min) / (grid.ny - 1);
    }

    // Build 3D interpolation points
    grid.interp_points.SetSize(3 * grid.npts);
    int idx = 0;
    for (int j = 0; j < grid.ny; ++j) {
        for (int i = 0; i < grid.nx; ++i) {
            Vector pt(3);
            pt = 0.0;
            pt(ax1) = grid.grid_x[i];
            pt(ax2) = grid.grid_y[j];
            pt(ax_fixed) = fixed_coord;
            grid.interp_points(idx++) = pt(0);
            grid.interp_points(idx++) = pt(1);
            grid.interp_points(idx++) = pt(2);
        }
    }

    grids_.push_back(std::move(grid));
}

// =============================================================================
// ShouldWrite
// =============================================================================

bool GMTWavefieldWriter::ShouldWrite(int step) const {
    return (step % interval_ == 0);
}

// =============================================================================
// Write
// =============================================================================

void GMTWavefieldWriter::Write(int step, real_t time,
                                const ParGridFunction* u,
                                const ParGridFunction* v,
                                const ParGridFunction* a,
                                int source_id) {
    for (const auto& grid : grids_) {
        auto write_field = [&](const ParGridFunction* gf, const std::string& name) {
            if (!gf) return;
            const int vdim = gf->VectorDim();

            // Resolve the component list for this field. Scalar fields
            // have only the trivial "magnitude" column; vector fields
            // walk `opts_.components`, dropping (with a warning) any
            // component that's out of range for the current vdim. The
            // warning text matches the old `InterpolateAndWrite` exactly
            // so users don't see a message regression.
            std::vector<int> comps_to_write;
            if (vdim == 1) {
                comps_to_write.push_back(-1);
            } else {
                for (int c : opts_.components) {
                    if (c >= vdim) {
                        if (rank_ == 0) {
                            MFEM_WARNING("GMTWavefieldWriter: component " << c
                                         << " is out of range for a vdim=" << vdim
                                         << " field ('" << name
                                         << "'); skipping. Valid: -1 (mag), 0.."
                                         << (vdim - 1) << ".");
                        }
                        continue;
                    }
                    comps_to_write.push_back(c);
                }
            }
            if (comps_to_write.empty()) return;

            // One interpolation sweep feeds every requested component —
            // that's the whole point of this refactor. For a DISP vector
            // with [-1, 0, 1] we used to do 3× sweeps × 3× GetVectorValue
            // calls per grid point; now it's 1× sweep × 1× GetVectorValue.
            std::vector<std::vector<real_t>> local_values_per_comp;
            std::vector<int> local_found;
            InterpolateField(grid, gf, comps_to_write,
                             local_values_per_comp, local_found);

            for (size_t k = 0; k < comps_to_write.size(); ++k) {
                ReduceAndWriteComponent(grid,
                                        local_values_per_comp[k],
                                        local_found,
                                        comps_to_write[k],
                                        name, step, vdim);
            }
        };

        if (opts_.write_displacement) write_field(u, "DISP");
        if (opts_.write_pressure)     write_field(u, "PS");
        if (opts_.write_velocity)     write_field(v, "VEL");
        if (opts_.write_acceleration) write_field(a, "ACC");
    }
}

// =============================================================================
// InterpolateField — Stage A: one sweep, many components
// =============================================================================

void GMTWavefieldWriter::InterpolateField(
    const GMTGrid& grid,
    const ParGridFunction* gf,
    const std::vector<int>& components,
    std::vector<std::vector<real_t>>& local_values_per_comp,
    std::vector<int>& local_found) {

    const auto& elem = grid.finder->GetElem();
    const auto& proc = grid.finder->GetProc();
    const auto& code = grid.finder->GetCode();
    const auto& ref_pos = grid.finder->GetReferencePosition();

    const int vdim = gf->VectorDim();
    const int ncomp = static_cast<int>(components.size());

    local_values_per_comp.assign(ncomp, std::vector<real_t>(grid.npts, 0.0));
    local_found.assign(grid.npts, 0);

    // Hoist the per-point scratch Vectors out of the inner loop — the
    // old code allocated both on every iteration (400k+ mallocs per
    // snapshot per field).
    Vector ref_pt(dim_);
    Vector val(vdim);  // size 1 when scalar, matches gf layout

    for (int p = 0; p < grid.npts; ++p) {
        if (code[p] == 2) continue;
        if (static_cast<int>(proc[p]) != rank_) continue;

        const int e = static_cast<int>(elem[p]);
        if (e < 0) continue;

        for (int d = 0; d < dim_; ++d) {
            ref_pt(d) = ref_pos(p * dim_ + d);
        }

        IntegrationPoint ip;
        ip.Set(ref_pt.GetData(), dim_);

        if (vdim == 1) {
            const real_t s = gf->GetValue(e, ip);
            // Single-component case: components must be {-1}; store the
            // scalar in every requested column (there is only one).
            for (int k = 0; k < ncomp; ++k) {
                local_values_per_comp[k][p] = s;
            }
        } else {
            // Vector case: one GetVectorValue call, then pick off each
            // requested component (-1 = magnitude, 0..vdim-1 = axis).
            gf->GetVectorValue(e, ip, val);
            for (int k = 0; k < ncomp; ++k) {
                local_values_per_comp[k][p] = ExtractComponent(val, components[k]);
            }
        }
        local_found[p] = 1;
    }
}

// =============================================================================
// ReduceAndWriteComponent — Stage B: MPI reduce + ASCII write
// =============================================================================

void GMTWavefieldWriter::ReduceAndWriteComponent(
    const GMTGrid& grid,
    const std::vector<real_t>& local_values,
    const std::vector<int>& local_found,
    int component,
    const std::string& field_name,
    int step,
    int vdim) {

    // Gather to rank 0 — one pair of reductions per component. Fusing
    // `found` across components is deferred to a follow-up PR so this
    // change stays minimal and the ASCII output stays byte-equal.
    std::vector<real_t> global_values(grid.npts, 0.0);
    std::vector<int> global_found(grid.npts, 0);
    MPI_Reduce(local_values.data(), global_values.data(), grid.npts,
               HYPRE_MPI_REAL, MPI_SUM, 0, comm_);
    MPI_Reduce(local_found.data(), global_found.data(), grid.npts,
               MPI_INT, MPI_SUM, 0, comm_);

    if (rank_ != 0) return;

    std::string gmt_dir = output_dir_ + "/wavefield/gmt";
    std::string comp_str = (vdim > 1) ? "_" + ComponentSuffix(component) : "";

    std::ostringstream filename;
    if (grid.name.empty()) {
        filename << gmt_dir << "/" << field_name << comp_str << "_"
                 << std::setfill('0') << std::setw(6) << step << ".xyz";
    } else {
        filename << gmt_dir << "/" << grid.name << "/" << field_name
                 << comp_str << "_"
                 << std::setfill('0') << std::setw(6) << step << ".xyz";
    }

    std::ofstream ofs(filename.str());
    if (!ofs.good()) return;

    ofs << std::scientific << std::setprecision(8);
    for (int j = 0; j < grid.ny; ++j) {
        for (int i = 0; i < grid.nx; ++i) {
            int p = j * grid.nx + i;
            ofs << grid.grid_x[i] << " " << grid.grid_y[j] << " ";
            if (global_found[p] > 0) {
                ofs << global_values[p];
            } else {
                ofs << "NaN";
            }
            ofs << "\n";
        }
    }
}

// =============================================================================
// ExtractComponent / ComponentSuffix
// =============================================================================

real_t GMTWavefieldWriter::ExtractComponent(const Vector& vec, int component) {
    if (component >= 0 && component < vec.Size()) {
        return vec(component);
    }
    // Magnitude
    real_t mag = 0.0;
    for (int i = 0; i < vec.Size(); ++i) {
        mag += vec(i) * vec(i);
    }
    return std::sqrt(mag);
}

std::string GMTWavefieldWriter::ComponentSuffix(int component) {
    switch (component) {
        case 0: return "x";
        case 1: return "y";
        case 2: return "z";
        default: return "mag";
    }
}

// =============================================================================
// Finalize
// =============================================================================

void GMTWavefieldWriter::Finalize() {
    // Clear grids for potential re-use in sequential source mode
    grids_.clear();
}

}  // namespace SEM
