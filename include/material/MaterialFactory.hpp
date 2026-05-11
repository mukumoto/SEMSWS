/**
 * @file MaterialFactory.hpp
 * @brief Centralized MaterialType → concrete Material class dispatch
 *
 * This is the single location in the codebase where a MaterialType enum is
 * mapped to a concrete derived Material class. Other modules must obtain
 * materials through CreateMaterial() rather than switching on MaterialType
 * themselves.
 */

#ifndef SEM_MATERIAL_FACTORY_HPP
#define SEM_MATERIAL_FACTORY_HPP

#include "material/MaterialBase.hpp"
#include <memory>

namespace mfem {
class ParFiniteElementSpace;
class IntegrationRule;
}

namespace SEM {

struct MaterialConfig;

namespace material {

/**
 * @brief Create a material instance from a MaterialConfig.
 *
 * Dispatches on (dim, cfg.material_type) to the appropriate derived class's
 * static FromConfig() factory, then applies ApplyAttenuationCorrection() so
 * that the returned material is ready for operator setup.
 *
 * @param cfg         Material configuration (contains material_type string)
 * @param fes_scalar  Scalar FE space (used by all FromConfig implementations)
 * @param ir          Integration rule (GLL)
 * @param dim         Spatial dimension (2 or 3)
 * @return Owning pointer to the instantiated material
 */
std::unique_ptr<MaterialBase> CreateMaterial(
    const MaterialConfig& cfg,
    mfem::ParFiniteElementSpace& fes_scalar,
    const mfem::IntegrationRule& ir,
    int dim);

}  // namespace material
}  // namespace SEM

#endif  // SEM_MATERIAL_FACTORY_HPP
