/**
 * @file StiffnessIntegratorFactory.cpp
 * @brief Implementation of the four free-function stiffness integrator
 *        factories declared in StiffnessIntegratorFactory.hpp.
 *
 * Each factory covers a single (physics, dimension) combination. Dispatch
 * inside the factory is only on MaterialType (pure vs. visco is read from
 * HasAttenuation()). The operator layer already carries the physics and
 * dimension in its own type, so it just calls the matching factory.
 */

#include "integ/StiffnessIntegratorFactory.hpp"
#include "material/isotropic_elastic/IsotropicElasticMaterial.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticMaterial.hpp"
#include "integ/SEMIsotropicElasticIntegrator.hpp"
#include "integ/SEMVisco_IsotropicElasticIntegrator.hpp"
#include "integ/SEMIsotropicAcousticIntegrator.hpp"
#include "integ/SEMVisco_IsotropicAcousticIntegrator.hpp"
#include "util/Profiler.hpp"
#include <climits>

namespace SEM {

using mfem::Vector;

// =============================================================================
// 2D Elastic
// =============================================================================

std::unique_ptr<mfem::BilinearFormIntegrator>
CreateElasticStiffnessIntegrator2D(const ElasticMaterialBase2D& material,
                                   mfem::ParFiniteElementSpace& fes,
                                   real_t dt)
{
    const int ne = material.NumElements();
    const int ngll = material.NumGLLx();
    const int total = ne * ngll * ngll;

    switch (material.GetType()) {
        case MaterialType::IsotropicElastic: {
            const auto& m = static_cast<const IsotropicElasticMaterial&>(material);

            if (material.HasAttenuation()) {
                // Viscoelastic: kappa/mu (already unrelaxed by ApplyAttenuationCorrection)
                Vector kappa_vec(total);
                Vector mu_vec(total);
                const real_t* k_src = m.Kappa().Data().HostRead();
                const real_t* mu_src = m.Mu().Data().HostRead();
                real_t* k_dst = kappa_vec.HostWrite();
                real_t* mu_dst = mu_vec.HostWrite();
                for (int i = 0; i < total; i++) {
                    k_dst[i] = k_src[i];
                    mu_dst[i] = mu_src[i];
                }

                auto integ = std::make_unique<SEMVisco_IsotropicElasticIntegrator2D>(
                    kappa_vec, mu_vec);
                {
                    PROFILE_REGION_GPU("Setup:StiffnessAssemblePA");
                    integ->AssemblePA(fes);
                }

                integ->GetAttenuation().EnableAttenuation(
                    ne, ngll, ngll,
                    m.Qkappa().Data(), m.Qmu().Data(),
                    integ->KappaVec(), integ->MuVec(),
                    m.AttenuationF0(), m.AttenuationNumUnits(), dt);
                integ->FinalizeMaterialParams();
                return integ;
            }

            // Pure elastic: lambda/mu
            Vector lambda_vec(total);
            Vector mu_vec(total);
            const real_t* lam_src = m.Lambda().Data().HostRead();
            const real_t* mu_src = m.Mu().Data().HostRead();
            real_t* lam_dst = lambda_vec.HostWrite();
            real_t* mu_dst = mu_vec.HostWrite();
            for (int i = 0; i < total; i++) {
                lam_dst[i] = lam_src[i];
                mu_dst[i] = mu_src[i];
            }

            auto integ = std::make_unique<SEMIsotropicElasticIntegrator2D>(
                lambda_vec, mu_vec);
            {
                PROFILE_REGION_GPU("Setup:StiffnessAssemblePA");
                integ->AssemblePA(fes);
            }
            return integ;
        }

        case MaterialType::AnisotropicElastic:
            MFEM_ABORT("Anisotropic elastic 2D integrator not yet implemented");

        default:
            MFEM_ABORT("Unsupported MaterialType for elastic 2D stiffness integrator");
    }
}

// =============================================================================
// 3D Elastic
// =============================================================================

std::unique_ptr<mfem::BilinearFormIntegrator>
CreateElasticStiffnessIntegrator3D(const ElasticMaterialBase3D& material,
                                   mfem::ParFiniteElementSpace& fes,
                                   real_t dt)
{
    const int ne = material.NumElements();
    const int ngll = material.NumGLLx();
    const int64_t total64 = (int64_t)ne * ngll * ngll * ngll;
    MFEM_VERIFY(total64 <= INT_MAX,
                "CreateElasticStiffnessIntegrator3D size (" << total64
                << ") exceeds int32 limit. Reduce elements per GPU (ne="
                << ne << ") or increase GPU count.");
    const int total = (int)total64;

    switch (material.GetType()) {
        case MaterialType::IsotropicElastic: {
            const auto& m = static_cast<const IsotropicElasticMaterial3D&>(material);

            if (material.HasAttenuation()) {
                Vector kappa_vec(total);
                Vector mu_vec(total);
                {
                    PROFILE_REGION("Setup:CopyKappaMu");
                    const real_t* k_src = m.Kappa().Data().HostRead();
                    const real_t* mu_src = m.Mu().Data().HostRead();
                    real_t* k_dst = kappa_vec.HostWrite();
                    real_t* mu_dst = mu_vec.HostWrite();
                    for (int i = 0; i < total; i++) {
                        k_dst[i] = k_src[i];
                        mu_dst[i] = mu_src[i];
                    }
                }

                auto integ = std::make_unique<SEMVisco_IsotropicElasticIntegrator3D>(
                    kappa_vec, mu_vec);
                {
                    PROFILE_REGION_GPU("Setup:StiffnessAssemblePA");
                    integ->AssemblePA(fes);
                }

                {
                    PROFILE_REGION("Setup:EnableAttenuation");
                    integ->GetAttenuation().EnableAttenuation(
                        ne, ngll, ngll, ngll,
                        m.Qkappa().Data(), m.Qmu().Data(),
                        integ->KappaVec(), integ->MuVec(),
                        m.AttenuationF0(), m.AttenuationNumUnits(), dt);
                }
                {
                    PROFILE_REGION("Setup:FinalizeMaterialParams");
                    integ->FinalizeMaterialParams();
                }
                return integ;
            }

            Vector lambda_vec(total);
            Vector mu_vec(total);
            const real_t* lam_src = m.Lambda().Data().HostRead();
            const real_t* mu_src = m.Mu().Data().HostRead();
            real_t* lam_dst = lambda_vec.HostWrite();
            real_t* mu_dst = mu_vec.HostWrite();
            for (int i = 0; i < total; i++) {
                lam_dst[i] = lam_src[i];
                mu_dst[i] = mu_src[i];
            }

            auto integ = std::make_unique<SEMIsotropicElasticIntegrator3D>(
                lambda_vec, mu_vec);
            {
                PROFILE_REGION_GPU("Setup:StiffnessAssemblePA");
                integ->AssemblePA(fes);
            }
            return integ;
        }

        case MaterialType::AnisotropicElastic:
            MFEM_ABORT("Anisotropic elastic 3D integrator not yet implemented");

        default:
            MFEM_ABORT("Unsupported MaterialType for elastic 3D stiffness integrator");
    }
}

// =============================================================================
// 2D Acoustic
// =============================================================================

std::unique_ptr<mfem::BilinearFormIntegrator>
CreateAcousticStiffnessIntegrator2D(const AcousticMaterialBase2D& material,
                                    mfem::ParFiniteElementSpace& fes,
                                    real_t dt)
{
    const int ne = material.NumElements();
    const int ngll = material.NumGLLx();
    const int total = ne * ngll * ngll;

    // Only IsotropicAcoustic exists. If new acoustic materials are
    // introduced later, add a switch on material.GetType() here.
    if (material.HasAttenuation()) {
        Vector inv_rho_vec(total);
        Vector kappa_vec(total);
        const real_t* ir_src = material.InvRho().Data().HostRead();
        const real_t* k_src = material.Kappa().Data().HostRead();
        real_t* ir_dst = inv_rho_vec.HostWrite();
        real_t* k_dst = kappa_vec.HostWrite();
        for (int i = 0; i < total; i++) {
            ir_dst[i] = ir_src[i];
            k_dst[i] = k_src[i];
        }

        auto integ = std::make_unique<SEMVisco_IsotropicAcousticIntegrator2D>(
            inv_rho_vec, kappa_vec);
        {
            PROFILE_REGION_GPU("Setup:StiffnessAssemblePA");
            integ->AssemblePA(fes);
        }

        integ->GetAttenuation().EnableAttenuation(
            ne, ngll, ngll,
            material.Qkappa().Data(),
            integ->KappaVec(),
            material.AttenuationF0(), material.AttenuationNumUnits(), dt);
        integ->FinalizeMaterialParams();
        return integ;
    }

    Vector inv_rho_vec(total);
    const real_t* src = material.InvRho().Data().HostRead();
    real_t* dst = inv_rho_vec.HostWrite();
    for (int i = 0; i < total; i++) {
        dst[i] = src[i];
    }

    auto integ = std::make_unique<SEMIsotropicAcousticIntegrator2D>(inv_rho_vec);
    {
        PROFILE_REGION_GPU("Setup:StiffnessAssemblePA");
        integ->AssemblePA(fes);
    }
    return integ;
}

// =============================================================================
// 3D Acoustic
// =============================================================================

std::unique_ptr<mfem::BilinearFormIntegrator>
CreateAcousticStiffnessIntegrator3D(const AcousticMaterialBase3D& material,
                                    mfem::ParFiniteElementSpace& fes,
                                    real_t dt)
{
    const int ne = material.NumElements();
    const int ngll = material.NumGLLx();
    const int64_t total64 = (int64_t)ne * ngll * ngll * ngll;
    MFEM_VERIFY(total64 <= INT_MAX,
                "CreateAcousticStiffnessIntegrator3D size (" << total64
                << ") exceeds int32 limit.");
    const int total = (int)total64;

    if (material.HasAttenuation()) {
        Vector inv_rho_vec(total);
        Vector kappa_vec(total);
        const real_t* ir_src = material.InvRho().Data().HostRead();
        const real_t* k_src = material.Kappa().Data().HostRead();
        real_t* ir_dst = inv_rho_vec.HostWrite();
        real_t* k_dst = kappa_vec.HostWrite();
        for (int i = 0; i < total; i++) {
            ir_dst[i] = ir_src[i];
            k_dst[i] = k_src[i];
        }

        auto integ = std::make_unique<SEMVisco_IsotropicAcousticIntegrator3D>(
            inv_rho_vec, kappa_vec);
        {
            PROFILE_REGION_GPU("Setup:StiffnessAssemblePA");
            integ->AssemblePA(fes);
        }

        integ->GetAttenuation().EnableAttenuation(
            ne, ngll, ngll, ngll,
            material.Qkappa().Data(),
            integ->KappaVec(),
            material.AttenuationF0(), material.AttenuationNumUnits(), dt);
        integ->FinalizeMaterialParams();
        return integ;
    }

    Vector inv_rho_vec(total);
    const real_t* src = material.InvRho().Data().HostRead();
    real_t* dst = inv_rho_vec.HostWrite();
    for (int i = 0; i < total; i++) {
        dst[i] = src[i];
    }

    auto integ = std::make_unique<SEMIsotropicAcousticIntegrator3D>(inv_rho_vec);
    {
        PROFILE_REGION_GPU("Setup:StiffnessAssemblePA");
        integ->AssemblePA(fes);
    }
    return integ;
}

}  // namespace SEM
