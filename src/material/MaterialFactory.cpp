/**
 * @file MaterialFactory.cpp
 * @brief MaterialType → concrete Material class dispatch
 */

#include "material/MaterialFactory.hpp"
#include "material/isotropic_elastic/IsotropicElasticMaterial.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticMaterial.hpp"
#include "config/ConfigTypes.hpp"
#include "common/Types.hpp"
#include <mfem.hpp>

namespace SEM {
namespace material {

std::unique_ptr<MaterialBase> CreateMaterial(
    const MaterialConfig& cfg,
    mfem::ParFiniteElementSpace& fes_scalar,
    const mfem::IntegrationRule& ir,
    int dim)
{
    const MaterialType type = StringToMaterialType(cfg.material_type);
    std::unique_ptr<MaterialBase> mat;

    if (dim == 2) {
        switch (type) {
            case MaterialType::IsotropicElastic:
                mat = IsotropicElasticMaterial::FromConfig(cfg, fes_scalar, ir);
                break;
            case MaterialType::IsotropicAcoustic:
                mat = IsotropicAcousticMaterial::FromConfig(cfg, fes_scalar, ir);
                break;
            case MaterialType::AnisotropicElastic:
                MFEM_ABORT("Anisotropic material not yet implemented for 2D");
        }
    } else if (dim == 3) {
        switch (type) {
            case MaterialType::IsotropicElastic:
                mat = IsotropicElasticMaterial3D::FromConfig(cfg, fes_scalar, ir);
                break;
            case MaterialType::IsotropicAcoustic:
                mat = IsotropicAcousticMaterial3D::FromConfig(cfg, fes_scalar, ir);
                break;
            case MaterialType::AnisotropicElastic:
                MFEM_ABORT("Anisotropic material not yet implemented for 3D");
        }
    } else {
        MFEM_ABORT("MaterialFactory: unsupported dimension " << dim);
    }

    MFEM_VERIFY(mat, "MaterialFactory: failed to construct material");

    // Apply attenuation correction (unrelaxed moduli) before operator setup.
    // Virtual dispatch; no-op when attenuation is disabled.
    mat->ApplyAttenuationCorrection();

    return mat;
}

}  // namespace material
}  // namespace SEM
