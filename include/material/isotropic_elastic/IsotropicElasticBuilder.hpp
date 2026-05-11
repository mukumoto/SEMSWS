/**
 * @file IsotropicElasticBuilder.hpp
 * @brief Builder for isotropic elastic materials
 *
 * Responsible for constructing IsotropicElasticMaterial from
 * IsotropicElasticInput populated by the Loader.
 *
 * Key responsibility:
 * - Automatically calls ApplyAttenuationCorrection() when attenuation is enabled
 * - This ensures correct unrelaxed moduli computation before operator setup
 *
 * Usage:
 *   auto input = IsotropicElasticLoader::Load(config, is_3d);
 *   auto material = IsotropicElasticBuilder<2>::Build(input, fes, ir);
 */

#ifndef SEM_ISOTROPIC_ELASTIC_BUILDER_HPP
#define SEM_ISOTROPIC_ELASTIC_BUILDER_HPP

#include "IsotropicElasticInput.hpp"
#include "material/isotropic_elastic/IsotropicElasticMaterial.hpp"
#include "material/InterpolatingCoefficient.hpp"
#include <mfem.hpp>
#include <memory>

namespace SEM {

using namespace mfem;

/**
 * @class IsotropicElasticBuilder
 * @brief Template class for building IsotropicElasticMaterial (2D/3D)
 *
 * @tparam Dim Dimension (2 or 3)
 *
 * The Builder:
 * 1. Creates material with appropriate dimensions
 * 2. Populates fields based on data source (constant/grid/by_attribute)
 * 3. Automatically applies attenuation correction when enabled
 */
template<int Dim>
class IsotropicElasticBuilder;

// 2D specialization
template<>
class IsotropicElasticBuilder<2> {
public:
    using Material = IsotropicElasticMaterial;

    /**
     * @brief Build material from loaded data
     *
     * @param data Loaded material data from IsotropicElasticLoader
     * @param fes Finite element space
     * @param ir Integration rule (GLL points)
     * @return Unique pointer to created material
     *
     * @note Automatically calls ApplyAttenuationCorrection() if attenuation is enabled
     */
    static std::unique_ptr<Material> Build(
        const IsotropicElasticInput& data,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

private:
    static std::unique_ptr<Material> BuildFromConstant(
        const IsotropicElasticConstantParams& params,
        const MaterialAttenuationConfig& attenuation,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromGrid(
        const IsotropicElasticInput& data,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromByAttribute(
        const std::vector<IsotropicElasticAttributeEntry>& entries,
        const MaterialAttenuationConfig& attenuation,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromByAttributeMixed(
        const std::vector<IsotropicElasticMixedAttributeEntry>& entries,
        const MaterialAttenuationConfig& attenuation,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromADIOS2(
        const IsotropicElasticInput& data,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static void GetMeshInfo(const ParFiniteElementSpace& fes, const IntegrationRule& ir,
                            int& ne, int& ngll);
};

// 3D specialization
template<>
class IsotropicElasticBuilder<3> {
public:
    using Material = IsotropicElasticMaterial3D;

    /**
     * @brief Build material from loaded data
     *
     * @param data Loaded material data from IsotropicElasticLoader
     * @param fes Finite element space
     * @param ir Integration rule (GLL points)
     * @return Unique pointer to created material
     *
     * @note Automatically calls ApplyAttenuationCorrection() if attenuation is enabled
     */
    static std::unique_ptr<Material> Build(
        const IsotropicElasticInput& data,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

private:
    static std::unique_ptr<Material> BuildFromConstant(
        const IsotropicElasticConstantParams& params,
        const MaterialAttenuationConfig& attenuation,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromGrid(
        const IsotropicElasticInput& data,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromByAttribute(
        const std::vector<IsotropicElasticAttributeEntry>& entries,
        const MaterialAttenuationConfig& attenuation,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromByAttributeMixed(
        const std::vector<IsotropicElasticMixedAttributeEntry>& entries,
        const MaterialAttenuationConfig& attenuation,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromADIOS2(
        const IsotropicElasticInput& data,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static void GetMeshInfo(const ParFiniteElementSpace& fes, const IntegrationRule& ir,
                            int& ne, int& ngll);
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ELASTIC_BUILDER_HPP
