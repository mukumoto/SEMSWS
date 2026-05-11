/**
 * @file MaterialWriter.cpp
 * @brief Implementation of MaterialWriter
 *
 * Dispatches on MaterialType for IsotropicElastic and IsotropicAcoustic.
 * Computes Vp, Vs, rho from internal Lame parameters and writes to
 * ParaView, GLVis, and GMT formats.
 */

#include "io/MaterialWriter.hpp"
#include "io/FieldVizWriter.hpp"
#include "material/ElasticMaterialBase.hpp"
#include "material/AcousticMaterialBase.hpp"
#include "material/MaterialField.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <memory>

#ifndef __linux__
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#endif

namespace SEM {

namespace {

bool CreateDir(const std::string& path) {
#ifdef __linux__
    return (mkdir(path.c_str(), 0755) == 0 || errno == EEXIST);
#else
    return (_mkdir(path.c_str()) == 0 || errno == EEXIST);
#endif
}

bool HasField(const std::vector<std::string>& fields, const std::string& name) {
    return std::find(fields.begin(), fields.end(), name) != fields.end();
}

// ============================================================================
// MaterialField computation helpers
// ============================================================================

/// Vp for 2D elastic: kappa = lambda + mu, so lambda+2mu = kappa+mu
/// Vp = sqrt((kappa + mu) / rho)
void ComputeVp2D(const MaterialField& kappa, const MaterialField& mu,
                  const MaterialField& rho, MaterialField& vp) {
    const auto* k = kappa.Data().HostRead();
    const auto* m = mu.Data().HostRead();
    const auto* r = rho.Data().HostRead();
    auto* out = vp.Data().HostWrite();
    for (int i = 0; i < kappa.Data().Size(); ++i) {
        out[i] = std::sqrt((k[i] + m[i]) / r[i]);
    }
}

/// Vp for 3D elastic: kappa = lambda + 2/3*mu, so lambda+2mu = kappa + 4/3*mu
/// Vp = sqrt((kappa + 4/3*mu) / rho)
void ComputeVp3D(const MaterialField3D& kappa, const MaterialField3D& mu,
                  const MaterialField3D& rho, MaterialField3D& vp) {
    const auto* k = kappa.Data().HostRead();
    const auto* m = mu.Data().HostRead();
    const auto* r = rho.Data().HostRead();
    auto* out = vp.Data().HostWrite();
    const real_t four_thirds = 4.0 / 3.0;
    for (int i = 0; i < kappa.Data().Size(); ++i) {
        out[i] = std::sqrt((k[i] + four_thirds * m[i]) / r[i]);
    }
}

/// Vs = sqrt(mu / rho)
template <typename Mu, typename Rho, typename Vs>
void ComputeVs(const Mu& mu, const Rho& rho, Vs& vs) {
    const auto* m = mu.Data().HostRead();
    const auto* r = rho.Data().HostRead();
    auto* out = vs.Data().HostWrite();
    for (int i = 0; i < mu.Data().Size(); ++i) {
        out[i] = std::sqrt(m[i] / r[i]);
    }
}

/// Vp for acoustic: Vp = sqrt(kappa * inv_rho)
template <typename Kappa, typename InvRho, typename Vp>
void ComputeVpAcoustic(const Kappa& kappa, const InvRho& inv_rho, Vp& vp) {
    const auto* k = kappa.Data().HostRead();
    const auto* ir = inv_rho.Data().HostRead();
    auto* out = vp.Data().HostWrite();
    for (int i = 0; i < kappa.Data().Size(); ++i) {
        out[i] = std::sqrt(k[i] * ir[i]);
    }
}

/// rho = 1 / inv_rho
template <typename InvRho, typename Rho>
void ComputeRhoFromInvRho(const InvRho& inv_rho, Rho& rho) {
    const auto* ir = inv_rho.Data().HostRead();
    auto* out = rho.Data().HostWrite();
    for (int i = 0; i < inv_rho.Data().Size(); ++i) {
        out[i] = 1.0 / ir[i];
    }
}

}  // anonymous namespace

// =============================================================================
// Write (main dispatch)
// =============================================================================

void MaterialWriter::Write(const MaterialBase& material,
                           ParFiniteElementSpace& fes,
                           const MaterialOutputConfig& config,
                           const std::string& output_dir,
                           MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    // Create output directories
    std::string mat_dir = output_dir + "/material";
    if (rank == 0) {
        CreateDir(output_dir);
        CreateDir(mat_dir);
    }
    MPI_Barrier(comm);

    // Collect material fields as ParGridFunctions
    std::vector<std::pair<std::string, std::unique_ptr<ParGridFunction>>> owned_fields;

    switch (material.GetType()) {
        case MaterialType::IsotropicElastic:
            CollectIsotropicElastic(material, fes, config, owned_fields);
            break;
        case MaterialType::IsotropicAcoustic:
            CollectIsotropicAcoustic(material, fes, config, owned_fields);
            break;
        default:
            if (rank == 0) {
                MFEM_WARNING("Material visualization not supported for this material type");
            }
            return;
    }

    // Build raw pointer pairs
    std::vector<std::pair<std::string, ParGridFunction*>> field_ptrs;
    for (auto& [name, gf] : owned_fields) {
        field_ptrs.emplace_back(name, gf.get());
    }

    if (field_ptrs.empty()) return;

    // Write in each requested format (delegated to FieldVizWriter so the
    // runtime and offline `sem_viz` paths share one implementation).
    for (const auto& fmt : config.formats) {
        if (fmt.type == "paraview") {
            FieldVizWriter::WriteParaView(output_dir, "material",
                                          field_ptrs, fmt, fes.GetParMesh());
        } else if (fmt.type == "glvis") {
            FieldVizWriter::WriteGLVis(output_dir, "material", field_ptrs);
        } else if (fmt.type == "gmt") {
            FieldVizWriter::WriteGMT(output_dir, "material",
                                     field_ptrs, fmt, *fes.GetParMesh(), comm);
        } else if (rank == 0) {
            MFEM_WARNING("Unknown material output format: " + fmt.type);
        }
    }
}

// =============================================================================
// CollectIsotropicElastic
// =============================================================================

void MaterialWriter::CollectIsotropicElastic(
    const MaterialBase& material,
    ParFiniteElementSpace& fes,
    const MaterialOutputConfig& config,
    std::vector<std::pair<std::string, std::unique_ptr<ParGridFunction>>>& fields) {

    int dim = fes.GetParMesh()->Dimension();

    if (dim == 2) {
        auto& mat = static_cast<const ElasticMaterialBase2D&>(material);

        if (HasField(config.fields, "vp")) {
            MaterialField vp_field(mat.Kappa());
            ComputeVp2D(mat.Kappa(), mat.Mu(), mat.Rho(), vp_field);
            auto gf = std::make_unique<ParGridFunction>(&fes);
            vp_field.ToParGridFunction(fes, *gf);
            fields.emplace_back("vp", std::move(gf));
        }
        if (HasField(config.fields, "vs")) {
            MaterialField vs_field(mat.Mu());
            ComputeVs(mat.Mu(), mat.Rho(), vs_field);
            auto gf = std::make_unique<ParGridFunction>(&fes);
            vs_field.ToParGridFunction(fes, *gf);
            fields.emplace_back("vs", std::move(gf));
        }
        if (HasField(config.fields, "rho")) {
            auto gf = std::make_unique<ParGridFunction>(&fes);
            mat.Rho().ToParGridFunction(fes, *gf);
            fields.emplace_back("rho", std::move(gf));
        }
        if (HasField(config.fields, "qkappa")) {
            if (material.HasAttenuation()) {
                auto gf = std::make_unique<ParGridFunction>(&fes);
                mat.Qkappa().ToParGridFunction(fes, *gf);
                fields.emplace_back("qkappa", std::move(gf));
            } else {
                MFEM_WARNING("qkappa requested but attenuation is disabled");
            }
        }
        if (HasField(config.fields, "qmu")) {
            if (material.HasAttenuation()) {
                auto gf = std::make_unique<ParGridFunction>(&fes);
                mat.Qmu().ToParGridFunction(fes, *gf);
                fields.emplace_back("qmu", std::move(gf));
            } else {
                MFEM_WARNING("qmu requested but attenuation is disabled");
            }
        }
    } else {
        // 3D
        auto& mat = static_cast<const ElasticMaterialBase3D&>(material);

        if (HasField(config.fields, "vp")) {
            MaterialField3D vp_field(mat.Kappa());
            ComputeVp3D(mat.Kappa(), mat.Mu(), mat.Rho(), vp_field);
            auto gf = std::make_unique<ParGridFunction>(&fes);
            vp_field.ToParGridFunction(fes, *gf);
            fields.emplace_back("vp", std::move(gf));
        }
        if (HasField(config.fields, "vs")) {
            MaterialField3D vs_field(mat.Mu());
            ComputeVs(mat.Mu(), mat.Rho(), vs_field);
            auto gf = std::make_unique<ParGridFunction>(&fes);
            vs_field.ToParGridFunction(fes, *gf);
            fields.emplace_back("vs", std::move(gf));
        }
        if (HasField(config.fields, "rho")) {
            auto gf = std::make_unique<ParGridFunction>(&fes);
            mat.Rho().ToParGridFunction(fes, *gf);
            fields.emplace_back("rho", std::move(gf));
        }
        if (HasField(config.fields, "qkappa")) {
            if (material.HasAttenuation()) {
                auto gf = std::make_unique<ParGridFunction>(&fes);
                mat.Qkappa().ToParGridFunction(fes, *gf);
                fields.emplace_back("qkappa", std::move(gf));
            } else {
                MFEM_WARNING("qkappa requested but attenuation is disabled");
            }
        }
        if (HasField(config.fields, "qmu")) {
            if (material.HasAttenuation()) {
                auto gf = std::make_unique<ParGridFunction>(&fes);
                mat.Qmu().ToParGridFunction(fes, *gf);
                fields.emplace_back("qmu", std::move(gf));
            } else {
                MFEM_WARNING("qmu requested but attenuation is disabled");
            }
        }
    }
}

// =============================================================================
// CollectIsotropicAcoustic
// =============================================================================

void MaterialWriter::CollectIsotropicAcoustic(
    const MaterialBase& material,
    ParFiniteElementSpace& fes,
    const MaterialOutputConfig& config,
    std::vector<std::pair<std::string, std::unique_ptr<ParGridFunction>>>& fields) {

    int dim = fes.GetParMesh()->Dimension();

    if (dim == 2) {
        auto& mat = static_cast<const AcousticMaterialBase2D&>(material);

        if (HasField(config.fields, "vp")) {
            MaterialField vp_field(mat.Kappa());
            ComputeVpAcoustic(mat.Kappa(), mat.InvRho(), vp_field);
            auto gf = std::make_unique<ParGridFunction>(&fes);
            vp_field.ToParGridFunction(fes, *gf);
            fields.emplace_back("vp", std::move(gf));
        }
        if (HasField(config.fields, "rho")) {
            MaterialField rho_field(mat.InvRho());
            ComputeRhoFromInvRho(mat.InvRho(), rho_field);
            auto gf = std::make_unique<ParGridFunction>(&fes);
            rho_field.ToParGridFunction(fes, *gf);
            fields.emplace_back("rho", std::move(gf));
        }
        if (HasField(config.fields, "vs")) {
            MFEM_WARNING("vs is not available for acoustic materials");
        }
        if (HasField(config.fields, "qkappa")) {
            if (material.HasAttenuation()) {
                auto gf = std::make_unique<ParGridFunction>(&fes);
                mat.Qkappa().ToParGridFunction(fes, *gf);
                fields.emplace_back("qkappa", std::move(gf));
            } else {
                MFEM_WARNING("qkappa requested but attenuation is disabled");
            }
        }
        if (HasField(config.fields, "qmu")) {
            MFEM_WARNING("qmu is not available for acoustic materials");
        }
    } else {
        // 3D
        auto& mat = static_cast<const AcousticMaterialBase3D&>(material);

        if (HasField(config.fields, "vp")) {
            MaterialField3D vp_field(mat.Kappa());
            ComputeVpAcoustic(mat.Kappa(), mat.InvRho(), vp_field);
            auto gf = std::make_unique<ParGridFunction>(&fes);
            vp_field.ToParGridFunction(fes, *gf);
            fields.emplace_back("vp", std::move(gf));
        }
        if (HasField(config.fields, "rho")) {
            MaterialField3D rho_field(mat.InvRho());
            ComputeRhoFromInvRho(mat.InvRho(), rho_field);
            auto gf = std::make_unique<ParGridFunction>(&fes);
            rho_field.ToParGridFunction(fes, *gf);
            fields.emplace_back("rho", std::move(gf));
        }
        if (HasField(config.fields, "vs")) {
            MFEM_WARNING("vs is not available for acoustic materials");
        }
        if (HasField(config.fields, "qkappa")) {
            if (material.HasAttenuation()) {
                auto gf = std::make_unique<ParGridFunction>(&fes);
                mat.Qkappa().ToParGridFunction(fes, *gf);
                fields.emplace_back("qkappa", std::move(gf));
            } else {
                MFEM_WARNING("qkappa requested but attenuation is disabled");
            }
        }
        if (HasField(config.fields, "qmu")) {
            MFEM_WARNING("qmu is not available for acoustic materials");
        }
    }
}


}  // namespace SEM
