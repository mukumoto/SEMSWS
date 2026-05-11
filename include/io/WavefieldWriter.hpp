/**
 * @file WavefieldWriter.hpp
 * @brief Wavefield output strategy classes for forward simulation
 *
 * Strategy pattern for wavefield output:
 * - GLVisWavefieldWriter: ParGridFunction::SaveAsOne() for GLVis visualization
 */

#ifndef SEM_WAVEFIELD_WRITER_HPP
#define SEM_WAVEFIELD_WRITER_HPP

#include <mfem.hpp>
#include <string>
#include <memory>

namespace SEM {


using namespace mfem;

// =============================================================================
// WavefieldWriter - Abstract Base Class
// =============================================================================

/**
 * @class WavefieldWriter
 * @brief Abstract interface for wavefield output strategies
 */
class WavefieldWriter {
public:
    virtual ~WavefieldWriter() = default;

    /**
     * @brief Initialize writer with mesh and simulation info
     * @param mesh Parallel mesh
     * @param total_steps Total simulation steps
     * @param comm MPI communicator
     */
    virtual void Init(ParMesh& mesh, int total_steps, MPI_Comm comm) = 0;

    /**
     * @brief Check if output should occur at this step
     * @param step Current time step
     * @return true if output should be written
     */
    virtual bool ShouldWrite(int step) const = 0;

    /**
     * @brief Write wavefield data
     * @param step Current time step
     * @param time Current simulation time
     * @param u Displacement field (may be nullptr)
     * @param v Velocity field (may be nullptr)
     * @param a Acceleration field (may be nullptr)
     * @param source_id Source ID for filename (-1 = no source info)
     */
    virtual void Write(int step, real_t time,
                       const ParGridFunction* u,
                       const ParGridFunction* v,
                       const ParGridFunction* a,
                       int source_id = -1) = 0;

    /**
     * @brief Finalize output (close files, flush buffers)
     */
    virtual void Finalize() = 0;

    /**
     * @brief Override the bounding box used to lay out regular grids.
     *
     * Coupled fluid-solid simulations hand each writer a submesh whose
     * own bbox covers only the elements on that side (e.g. just the
     * letter-shaped fluid inclusions). For grid-based outputs (GMT)
     * that means the fluid and solid xyz files end up on DIFFERENT
     * grids, which breaks overlay visualisations. Calling this with
     * the parent-mesh bbox BEFORE `Init()` makes both writers share a
     * common interpolation grid — points outside the submesh are
     * padded with NaN as usual.
     *
     * Default: no-op (writers that don't use a regular grid ignore it).
     */
    virtual void SetBoundingBoxOverride(const mfem::Vector& bb_min,
                                         const mfem::Vector& bb_max) {
        (void)bb_min; (void)bb_max;
    }
};

// =============================================================================
// GLVisWavefieldWriter - Implemented
// =============================================================================

/**
 * @class GLVisWavefieldWriter
 * @brief GLVis visualization output using ParGridFunction::SaveAsOne()
 *
 * Output format:
 * - Mesh: output_dir/mesh.mesh (saved once on first write)
 * - Wavefield: output_dir/wavefield/u_NNNNNN.gf (per step)
 *
 * Uses MFEM's SaveAsOne which gathers data to rank 0 for serial output.
 */
class GLVisWavefieldWriter : public WavefieldWriter {
public:
    /**
     * @brief Construct GLVis writer
     * @param output_dir Output directory path
     * @param interval Output interval in steps (default 100)
     */
    GLVisWavefieldWriter(const std::string& output_dir, int interval = 100);

    void Init(ParMesh& mesh, int total_steps, MPI_Comm comm) override;
    bool ShouldWrite(int step) const override;
    void Write(int step, real_t time,
               const ParGridFunction* u,
               const ParGridFunction* v,
               const ParGridFunction* a,
               int source_id = -1) override;
    void Finalize() override;

private:
    std::string output_dir_;
    int interval_;
    int rank_ = 0;
    bool mesh_saved_ = false;
    ParMesh* mesh_ = nullptr;
};

}  // namespace SEM

#endif  // SEM_WAVEFIELD_WRITER_HPP
