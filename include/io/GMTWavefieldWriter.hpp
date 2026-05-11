/**
 * @file GMTWavefieldWriter.hpp
 * @brief GMT-compatible ASCII xyz wavefield output
 *
 * Interpolates wavefield from GLL points onto a regular grid
 * for GMT visualization. Supports 2D (full domain) and 3D (cross-sections).
 * Points outside the mesh are written as NaN.
 */

#ifndef SEM_GMT_WAVEFIELD_WRITER_HPP
#define SEM_GMT_WAVEFIELD_WRITER_HPP

#include "io/WavefieldWriter.hpp"
#include "util/PointFinder.hpp"
#include "config/ConfigTypes.hpp"
#include <mfem.hpp>
#include <string>
#include <memory>
#include <vector>

namespace SEM {

using namespace mfem;

/**
 * @brief Information for a single interpolation grid (2D plane)
 *
 * For 2D simulations: one grid covering the full domain.
 * For 3D simulations: one grid per cross-section slice.
 */
struct GMTGrid {
    std::string name;             ///< Subdirectory name (e.g., "xy_z250.0")
    int nx, ny;                   ///< Grid dimensions
    real_t x_min, x_max;         ///< Grid extent in first axis
    real_t y_min, y_max;         ///< Grid extent in second axis
    std::string axis1, axis2;     ///< Axis labels (e.g., "x", "z")
    std::vector<real_t> grid_x;  ///< 1D coordinates of first axis
    std::vector<real_t> grid_y;  ///< 1D coordinates of second axis

    std::unique_ptr<PointFinder> finder;
    Vector interp_points;         ///< Flattened 3D coords for FindPoints [dim*npts]
    int npts;                     ///< Total grid points (nx*ny)

    /// Write grid_info.txt header file for plotting scripts
    void WriteGridInfo(const std::string& dir) const;
};

/**
 * @class GMTWavefieldWriter
 * @brief GMT xyz output strategy for wavefield snapshots
 *
 * Output format: ASCII "x y value" per line (2D) or "axis1 axis2 value" (3D slice).
 *
 * Output structure:
 *   2D: {output_dir}/wavefield/gmt/{field}_{step:06d}.xyz
 *   3D: {output_dir}/wavefield/gmt/{slice_name}/{field}_{step:06d}.xyz
 */
class GMTWavefieldWriter : public WavefieldWriter {
public:
    struct Options {
        std::array<int,2> resolution = {100, 100};
        std::vector<int> components = {-1}; ///< -1=magnitude, 0=x, 1=y, 2=z
        GMTCrossSections cross_sections;
        bool write_displacement = true;
        bool write_velocity = false;
        bool write_acceleration = false;
        bool write_pressure = false;
    };

    GMTWavefieldWriter(const std::string& output_dir, int interval,
                       const Options& opts);

    void Init(ParMesh& mesh, int total_steps, MPI_Comm comm) override;
    bool ShouldWrite(int step) const override;
    void Write(int step, real_t time,
               const ParGridFunction* u,
               const ParGridFunction* v,
               const ParGridFunction* a,
               int source_id = -1) override;
    void Finalize() override;

    void SetBoundingBoxOverride(const Vector& bb_min,
                                const Vector& bb_max) override;

private:
    void Setup2DGrid(ParMesh& mesh);
    void Setup3DGrids(ParMesh& mesh);
    void SetupSliceGrid(const std::string& name,
                        const std::string& plane,
                        real_t fixed_coord,
                        const Vector& bb_min, const Vector& bb_max);

    // Step 1 — interpolate the field onto the grid once per call, filling
    // one column per requested component. For a solid DISP with
    // components=[-1, 0, 1] this replaces 3× full grid sweeps (the old
    // per-component `InterpolateAndWrite`) with a single sweep that calls
    // `gf->GetVectorValue` once per grid point and extracts every
    // requested component from the same result.
    void InterpolateField(const GMTGrid& grid,
                          const ParGridFunction* gf,
                          const std::vector<int>& components,
                          std::vector<std::vector<real_t>>& local_values_per_comp,
                          std::vector<int>& local_found);

    // Step 2 — MPI-reduce the per-component local buffer and write the
    // ASCII `.xyz` file on rank 0. Byte-equivalent to the write portion
    // of the old `InterpolateAndWrite`.
    void ReduceAndWriteComponent(const GMTGrid& grid,
                                 const std::vector<real_t>& local_values,
                                 const std::vector<int>& local_found,
                                 int component,
                                 const std::string& field_name,
                                 int step,
                                 int vdim);

    static real_t ExtractComponent(const Vector& vec, int component);
    static std::string ComponentSuffix(int component);

    std::string output_dir_;
    int interval_;
    Options opts_;
    int rank_ = 0;
    int dim_ = 0;
    MPI_Comm comm_;

    // Optional parent-mesh bounding-box override (see base class doc).
    // When set, Setup2DGrid / Setup3DGrids lay out grids over this
    // rectangle instead of the submesh's own bbox; points outside the
    // submesh fall through to NaN via PointFinder.
    bool has_bbox_override_ = false;
    Vector bbox_min_override_, bbox_max_override_;

    std::vector<GMTGrid> grids_;
};

}  // namespace SEM

#endif  // SEM_GMT_WAVEFIELD_WRITER_HPP
