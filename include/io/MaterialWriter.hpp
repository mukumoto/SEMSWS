/**
 * @file MaterialWriter.hpp
 * @brief Material field visualization output
 *
 * Writes material properties (Vp, Vs, rho, Qkappa, Qmu) in various formats.
 * Called once at simulation startup, not per-timestep.
 * Supports IsotropicElastic and IsotropicAcoustic materials.
 */

#ifndef SEM_MATERIAL_WRITER_HPP
#define SEM_MATERIAL_WRITER_HPP

#include "config/ConfigTypes.hpp"
#include "material/MaterialBase.hpp"
#include <mfem.hpp>
#include <string>
#include <vector>
#include <memory>

namespace SEM {

using namespace mfem;

class ElasticMaterialBase2D;
class ElasticMaterialBase3D;
class AcousticMaterialBase2D;
class AcousticMaterialBase3D;

/**
 * @class MaterialWriter
 * @brief Writes material fields to ParaView, GLVis, or GMT format
 *
 * Dispatches on MaterialType enum for extensibility:
 * - IsotropicElastic: vp, vs, rho, qkappa, qmu
 * - IsotropicAcoustic: vp, rho, qkappa
 * - Others: warning and skip
 */
class MaterialWriter {
public:
    /**
     * @brief Write material fields based on configuration
     * @param material Material object (elastic or acoustic)
     * @param fes Scalar finite element space for material fields (vdim=1)
     * @param config Material output configuration
     * @param output_dir Base output directory
     * @param comm MPI communicator
     */
    static void Write(const MaterialBase& material,
                      ParFiniteElementSpace& fes,
                      const MaterialOutputConfig& config,
                      const std::string& output_dir,
                      MPI_Comm comm);

private:
    /// Collect fields for IsotropicElastic (2D and 3D)
    static void CollectIsotropicElastic(
        const MaterialBase& material,
        ParFiniteElementSpace& fes,
        const MaterialOutputConfig& config,
        std::vector<std::pair<std::string, std::unique_ptr<ParGridFunction>>>& fields);

    /// Collect fields for IsotropicAcoustic (2D and 3D)
    static void CollectIsotropicAcoustic(
        const MaterialBase& material,
        ParFiniteElementSpace& fes,
        const MaterialOutputConfig& config,
        std::vector<std::pair<std::string, std::unique_ptr<ParGridFunction>>>& fields);
};

}  // namespace SEM

#endif  // SEM_MATERIAL_WRITER_HPP
