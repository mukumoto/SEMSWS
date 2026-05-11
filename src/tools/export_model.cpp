/**
 * @file export_model.cpp
 * @brief Materialize a YAML-described material once and write per-field BPs
 *
 * Reads a SEMSWS YAML config, loads the mesh and the material on its GLL
 * points, and writes vp.bp / vs.bp / rho.bp (plus qkappa.bp / qmu.bp when
 * attenuation is enabled) into an output directory.
 *
 * Intended use: pre-pass before a multi-shot forward / FWI run when the
 * user-provided material is `format: grid` or `by_attribute_mixed` with a
 * `grid` sub-mode. Running this once into a shared directory lets every
 * subsequent shot consume the same BPs via `format: adios2`, avoiding N
 * concurrent ASCII grid reads.
 *
 * Usage:
 *   mpirun -np N ./semsws_export_model -config <config.yaml> -out <out_dir>
 *
 * Notes on parallelism:
 *   - The BPs encode the writer-side `nprocs` and per-rank ne_local. Any
 *     run that later reads them with `format: adios2` must use the SAME
 *     number of MPI ranks as this export step (SEMSWS LoadFieldBP enforces).
 *   - The mesh partitioning is otherwise deterministic with MFEM's default
 *     partitioner, so per-rank ne_local matches automatically.
 */

#include <mfem.hpp>
#include "config/YamlConfig.hpp"
#include "config/ConfigLoaders.hpp"
#include "config/ConfigTypes.hpp"
#include "common/Types.hpp"
#include "material/MaterialBase.hpp"
#include "material/ElasticMaterialBase.hpp"
#include "material/AcousticMaterialBase.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticMaterial.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticIO.hpp"
#include "material/isotropic_elastic/IsotropicElasticMaterial.hpp"
#include "material/isotropic_elastic/IsotropicElasticIO.hpp"

#include <iostream>
#include <string>
#include <sys/stat.h>

using namespace mfem;
using namespace SEM;

namespace {

void EnsureDir(const std::string& dir) {
    // Create parent directories. Idempotent. No-op when dir already exists.
    struct stat st {};
    if (stat(dir.c_str(), &st) == 0) return;
    // mkdir is enough for the leaf; callers should pass an absolute or
    // pre-existing-parent path.
    mkdir(dir.c_str(), 0755);
}

template<int Dim>
int RunExport(const YamlConfig& config, const std::string& out_dir) {
    int order = config.GetOrder();

    // --- Mesh ---
    std::unique_ptr<ParMesh> pmesh(LoadParMesh(config, MPI_COMM_WORLD));
    pmesh->SetCurvature(order, true, Dim, Ordering::byNODES);

    // --- FES + GLL integration rule ---
    H1_FECollection fec(order, Dim);
    ParFiniteElementSpace fes(pmesh.get(), &fec);

    IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);
    const int exact = 2 * order - 1;
    IntegrationRule ir_1d = gll_rules.Get(Geometry::SEGMENT, exact);
    IntegrationRule ir;
    if constexpr (Dim == 2) {
        ir = IntegrationRule(ir_1d, ir_1d);
    } else {
        ir = IntegrationRule(ir_1d, ir_1d, ir_1d);
    }

    // --- Material ---
    MaterialConfig mat_config = LoadMaterialConfig(config);
    MaterialType mat_type = StringToMaterialType(mat_config.material_type);

    std::unique_ptr<MaterialBase> material;
    if constexpr (Dim == 2) {
        if (mat_type == MaterialType::IsotropicElastic) {
            material = IsotropicElasticMaterial::FromConfig(mat_config, fes, ir);
        } else if (mat_type == MaterialType::IsotropicAcoustic) {
            material = IsotropicAcousticMaterial::FromConfig(mat_config, fes, ir);
        } else {
            if (Mpi::Root()) {
                std::cerr << "Error: unsupported material.type for export: "
                          << mat_config.material_type << std::endl;
            }
            return 1;
        }
    } else {
        if (mat_type == MaterialType::IsotropicElastic) {
            material = IsotropicElasticMaterial3D::FromConfig(mat_config, fes, ir);
        } else if (mat_type == MaterialType::IsotropicAcoustic) {
            material = IsotropicAcousticMaterial3D::FromConfig(mat_config, fes, ir);
        } else {
            if (Mpi::Root()) {
                std::cerr << "Error: unsupported material.type for export: "
                          << mat_config.material_type << std::endl;
            }
            return 1;
        }
    }

    // --- Output directory ---
    if (Mpi::Root()) {
        EnsureDir(out_dir);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // --- Dispatch to the matching ExportMaterialBP overload ---
    if (mat_type == MaterialType::IsotropicAcoustic) {
        if constexpr (Dim == 2) {
            const auto& m = static_cast<const AcousticMaterialBase2D&>(*material);
            ExportAcousticMaterialBP(m, out_dir, "", MPI_COMM_WORLD);
        } else {
            const auto& m = static_cast<const AcousticMaterialBase3D&>(*material);
            ExportAcousticMaterialBP(m, out_dir, "", MPI_COMM_WORLD);
        }
    } else {
        if constexpr (Dim == 2) {
            const auto& m = static_cast<const ElasticMaterialBase2D&>(*material);
            ExportElasticMaterialBP(m, out_dir, "", MPI_COMM_WORLD);
        } else {
            const auto& m = static_cast<const ElasticMaterialBase3D&>(*material);
            ExportElasticMaterialBP(m, out_dir, "", MPI_COMM_WORLD);
        }
    }

    if (Mpi::Root()) {
        std::cout << "[export_model] wrote BP files to: " << out_dir << std::endl;
    }
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    Hypre::Init();

    std::string config_file;
    std::string out_dir;
    OptionsParser args(argc, argv);
    args.AddOption(&config_file, "-config", "--config",
                   "Configuration file (YAML); same schema as the semsws binary");
    args.AddOption(&out_dir, "-out", "--out-dir",
                   "Output directory for vp.bp / vs.bp / rho.bp [/qkappa.bp / qmu.bp]");
    args.Parse();

    if (!args.Good() || config_file.empty() || out_dir.empty()) {
        if (Mpi::Root()) {
            std::cout << "\nUsage: " << argv[0]
                      << " -config <config.yaml> -out <out_dir>\n\n";
            args.PrintUsage(std::cout);
        }
        return 1;
    }

    std::unique_ptr<YamlConfig> config;
    try {
        config = std::make_unique<YamlConfig>(config_file);
    } catch (const std::exception& e) {
        if (Mpi::Root()) {
            std::cerr << "Error reading config: " << e.what() << std::endl;
        }
        return 1;
    }
    if (!config->IsValid()) {
        if (Mpi::Root()) {
            std::cerr << "Error validating config: "
                      << config->GetValidationError() << std::endl;
        }
        return 1;
    }

    int rc = (config->GetDimension() == 2)
                 ? RunExport<2>(*config, out_dir)
                 : RunExport<3>(*config, out_dir);
    return rc;
}
