/**
 * @file ParaViewWavefieldWriter.hpp
 * @brief ParaView VTK wavefield output using MFEM's ParaViewDataCollection
 *
 * Outputs wavefield snapshots as .pvd/.vtu files for ParaView visualization.
 * Always uses high-order Lagrange output (requires ParaView 5.5+).
 */

#ifndef SEM_PARAVIEW_WAVEFIELD_WRITER_HPP
#define SEM_PARAVIEW_WAVEFIELD_WRITER_HPP

#include "io/WavefieldWriter.hpp"
#include <mfem.hpp>
#include <string>
#include <memory>

namespace SEM {

using namespace mfem;

/**
 * @class ParaViewWavefieldWriter
 * @brief ParaView VTK output strategy for wavefield snapshots
 *
 * Uses MFEM's ParaViewDataCollection to produce .pvd + .vtu files.
 * Creates owned copies of GridFunctions because RegisterField() requires
 * non-const pointers.
 *
 * Output structure:
 *   {output_dir}/wavefield/paraview/wavefield.pvd
 *   {output_dir}/wavefield/paraview/Cycle{N}/proc{rank}.vtu
 */
class ParaViewWavefieldWriter : public WavefieldWriter {
public:
    struct Options {
        int refinement = 1;              ///< Element subdivision level
        std::string data_format = "binary32"; ///< "ascii", "binary"(64bit), "binary32"(32bit)
        int compression = -1;            ///< zlib compression (-1=default)
        bool write_displacement = true;
        bool write_velocity = false;
        bool write_acceleration = false;
        bool write_pressure = false;
    };

    /**
     * @brief Construct ParaView writer
     * @param output_dir Output directory path
     * @param interval Output interval in steps
     * @param opts Format options
     */
    ParaViewWavefieldWriter(const std::string& output_dir, int interval,
                            const Options& opts);

    void Init(ParMesh& mesh, int total_steps, MPI_Comm comm) override;
    bool ShouldWrite(int step) const override;
    void Write(int step, real_t time,
               const ParGridFunction* u,
               const ParGridFunction* v,
               const ParGridFunction* a,
               int source_id = -1) override;
    void Finalize() override;

private:
    void InitCollection(const ParGridFunction* u);
    static VTKFormat ParseVTKFormat(const std::string& fmt);

    std::string output_dir_;
    int interval_;
    Options opts_;
    int rank_ = 0;

    std::unique_ptr<ParaViewDataCollection> pv_dc_;
    bool initialized_ = false;

    // Owned copies for RegisterField (non-const requirement)
    std::unique_ptr<ParGridFunction> u_copy_;
    std::unique_ptr<ParGridFunction> v_copy_;
    std::unique_ptr<ParGridFunction> a_copy_;
};

}  // namespace SEM

#endif  // SEM_PARAVIEW_WAVEFIELD_WRITER_HPP
