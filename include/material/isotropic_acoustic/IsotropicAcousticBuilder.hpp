/**
 * @file IsotropicAcousticBuilder.hpp
 * @brief Builder for isotropic acoustic materials
 *
 * Responsible for constructing IsotropicAcousticMaterial from
 * IsotropicAcousticInput populated by the Loader.
 *
 * Note: Acoustic materials store kappa and inv_rho (not vp and rho directly).
 */

#ifndef SEM_ISOTROPIC_ACOUSTIC_BUILDER_HPP
#define SEM_ISOTROPIC_ACOUSTIC_BUILDER_HPP

#include "IsotropicAcousticInput.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticMaterial.hpp"
#include "material/InterpolatingCoefficient.hpp"
#include <mfem.hpp>
#include <memory>

namespace SEM {

using namespace mfem;

/**
 * @class IsotropicAcousticBuilder
 * @brief Template class for building IsotropicAcousticMaterial (2D/3D)
 *
 * @tparam Dim Dimension (2 or 3)
 */
template<int Dim>
class IsotropicAcousticBuilder;

// 2D specialization
template<>
class IsotropicAcousticBuilder<2> {
public:
    using Material = IsotropicAcousticMaterial;

    /**
     * @brief Build acoustic material from loaded data
     *
     * @param data Loaded material data from IsotropicAcousticLoader
     * @param fes Finite element space
     * @param ir Integration rule (GLL points)
     * @return Unique pointer to created material
     *
     * @note Automatically calls ApplyAttenuationCorrection() if attenuation is enabled
     */
    static std::unique_ptr<Material> Build(
        const IsotropicAcousticInput& data,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

private:
    static std::unique_ptr<Material> BuildFromConstant(
        const IsotropicAcousticConstantParams& params,
        const MaterialAttenuationConfig& attenuation,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromGrid(
        const IsotropicAcousticInput& data,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromByAttribute(
        const std::vector<IsotropicAcousticAttributeEntry>& entries,
        const MaterialAttenuationConfig& attenuation,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromByAttributeMixed(
        const std::vector<IsotropicAcousticMixedAttributeEntry>& entries,
        const MaterialAttenuationConfig& attenuation,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromADIOS2(
        const IsotropicAcousticInput& data,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static void GetMeshInfo(const ParFiniteElementSpace& fes, const IntegrationRule& ir,
                            int& ne, int& ngll);
};

// 3D specialization
template<>
class IsotropicAcousticBuilder<3> {
public:
    using Material = IsotropicAcousticMaterial3D;

    /**
     * @brief Build acoustic material from loaded data
     *
     * @param data Loaded material data from IsotropicAcousticLoader
     * @param fes Finite element space
     * @param ir Integration rule (GLL points)
     * @return Unique pointer to created material
     *
     * @note Automatically calls ApplyAttenuationCorrection() if attenuation is enabled
     */
    static std::unique_ptr<Material> Build(
        const IsotropicAcousticInput& data,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

private:
    static std::unique_ptr<Material> BuildFromConstant(
        const IsotropicAcousticConstantParams& params,
        const MaterialAttenuationConfig& attenuation,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromGrid(
        const IsotropicAcousticInput& data,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromByAttribute(
        const std::vector<IsotropicAcousticAttributeEntry>& entries,
        const MaterialAttenuationConfig& attenuation,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromByAttributeMixed(
        const std::vector<IsotropicAcousticMixedAttributeEntry>& entries,
        const MaterialAttenuationConfig& attenuation,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static std::unique_ptr<Material> BuildFromADIOS2(
        const IsotropicAcousticInput& data,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    static void GetMeshInfo(const ParFiniteElementSpace& fes, const IntegrationRule& ir,
                            int& ne, int& ngll);
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ACOUSTIC_BUILDER_HPP
