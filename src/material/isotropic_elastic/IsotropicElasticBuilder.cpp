/**
 * @file IsotropicElasticBuilder.cpp
 * @brief Implementation of IsotropicElasticBuilder
 */

#include "material/isotropic_elastic/IsotropicElasticBuilder.hpp"
#include "material/MaterialUtils.hpp"
#include <mfem.hpp>
#include <map>
#include <cmath>

namespace SEM {

// =============================================================================
// 2D Builder Implementation
// =============================================================================

std::unique_ptr<IsotropicElasticMaterial> IsotropicElasticBuilder<2>::Build(
    const IsotropicElasticInput& data,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    MFEM_VERIFY(data.IsValid(), "Invalid IsotropicElasticInput");

    std::unique_ptr<IsotropicElasticMaterial> material;

    switch (data.source) {
        case DataSource::Constant:
            material = BuildFromConstant(*data.constant, data.attenuation, fes, ir);
            break;
        case DataSource::Grid:
            material = BuildFromGrid(data, fes, ir);
            break;
        case DataSource::ByAttribute:
            material = BuildFromByAttribute(*data.by_attribute, data.attenuation, fes, ir);
            break;
        case DataSource::ByAttributeMixed:
            material = BuildFromByAttributeMixed(*data.by_attribute_mixed, data.attenuation, fes, ir);
            break;
        case DataSource::ADIOS2:
            material = BuildFromADIOS2(data, fes, ir);
            break;
    }

    // Automatically apply attenuation correction
    if (data.attenuation.enabled && material->HasAttenuation()) {
        material->ApplyAttenuationCorrection();
    }

    return material;
}

std::unique_ptr<IsotropicElasticMaterial> IsotropicElasticBuilder<2>::BuildFromConstant(
    const IsotropicElasticConstantParams& params,
    const MaterialAttenuationConfig& attenuation,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    auto [lambda, mu] = VelocityToLame(params.vp, params.vs, params.rho);

    auto material = std::make_unique<IsotropicElasticMaterial>(ne, ngll, ngll);
    material->Lambda().SetConstant(lambda);
    material->Mu().SetConstant(mu);
    material->Rho().SetConstant(params.rho);
    material->ComputeKappaFromLambdaMu();

    if (attenuation.enabled) {
        material->InitializeAttenuationConstant(
            params.qkappa, params.qmu,
            attenuation.f0, attenuation.n_units);
    }

    return material;
}

std::unique_ptr<IsotropicElasticMaterial> IsotropicElasticBuilder<2>::BuildFromGrid(
    const IsotropicElasticInput& data,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    MFEM_VERIFY(data.grid_2d.has_value(), "2D grid data not available");
    const auto& grid = *data.grid_2d;

    // Create interpolating coefficients
    InterpolatingCoefficient2D vp_coef(
        grid.vp, grid.nx, grid.ny, grid.dx, grid.dy, grid.x0, grid.y0);
    InterpolatingCoefficient2D vs_coef(
        grid.vs, grid.nx, grid.ny, grid.dx, grid.dy, grid.x0, grid.y0);
    InterpolatingCoefficient2D rho_coef(
        grid.rho, grid.nx, grid.ny, grid.dx, grid.dy, grid.x0, grid.y0);

    auto material = std::make_unique<IsotropicElasticMaterial>(
        vp_coef, vs_coef, rho_coef, fes, ir);

    // Initialize attenuation from grid data
    if (data.attenuation.enabled && grid.has_Q) {
        InterpolatingCoefficient2D qkappa_coef(
            grid.Qkappa, grid.nx, grid.ny, grid.dx, grid.dy, grid.x0, grid.y0);
        InterpolatingCoefficient2D qmu_coef(
            grid.Qmu, grid.nx, grid.ny, grid.dx, grid.dy, grid.x0, grid.y0);
        material->InitializeAttenuationFromCoefficient(
            qkappa_coef, qmu_coef, fes, ir,
            data.attenuation.f0, data.attenuation.n_units);
    }

    return material;
}

std::unique_ptr<IsotropicElasticMaterial> IsotropicElasticBuilder<2>::BuildFromByAttribute(
    const std::vector<IsotropicElasticAttributeEntry>& entries,
    const MaterialAttenuationConfig& attenuation,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    // Build attribute map
    std::map<int, IsotropicElasticAttributeEntry> attr_map;
    for (const auto& entry : entries) {
        attr_map[entry.attribute] = entry;
    }

    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    auto material = std::make_unique<IsotropicElasticMaterial>(ne, ngll, ngll);

    // Allocate Q fields if attenuation is enabled
    if (attenuation.enabled) {
        material->AllocateAttenuationFields(attenuation.f0, attenuation.n_units);
    }

    // Get ViewWrite for each field
    auto lambda_view = material->Lambda().ViewHostWrite();
    auto mu_view = material->Mu().ViewHostWrite();
    auto rho_view = material->Rho().ViewHostWrite();

    Mesh* mesh = fes.GetMesh();

    // Fill each element based on its attribute
    for (int e = 0; e < ne; e++) {
        int attr = mesh->GetAttribute(e);
        auto it = attr_map.find(attr);
        MFEM_VERIFY(it != attr_map.end(),
            "No material defined for attribute " << attr);

        const auto& entry = it->second;
        auto [lambda, mu] = VelocityToLame(entry.vp, entry.vs, entry.rho);

        for (int j = 0; j < ngll; j++) {
            for (int i = 0; i < ngll; i++) {
                lambda_view(i, j, e) = lambda;
                mu_view(i, j, e) = mu;
                rho_view(i, j, e) = entry.rho;
            }
        }

        // Set Q values if attenuation enabled
        if (attenuation.enabled) {
            auto qkappa_view = material->Qkappa().ViewHostWrite();
            auto qmu_view = material->Qmu().ViewHostWrite();
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    qkappa_view(i, j, e) = entry.Qkappa;
                    qmu_view(i, j, e) = entry.Qmu;
                }
            }
        }
    }

    material->ComputeKappaFromLambdaMu();
    return material;
}

std::unique_ptr<IsotropicElasticMaterial> IsotropicElasticBuilder<2>::BuildFromByAttributeMixed(
    const std::vector<IsotropicElasticMixedAttributeEntry>& entries,
    const MaterialAttenuationConfig& attenuation,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    // Build attribute map
    std::map<int, const IsotropicElasticMixedAttributeEntry*> attr_map;
    for (const auto& entry : entries) {
        attr_map[entry.attribute] = &entry;
    }

    // Create interpolating coefficients for heterogeneous attributes
    std::map<int, std::unique_ptr<InterpolatingCoefficient2D>> vp_coefs;
    std::map<int, std::unique_ptr<InterpolatingCoefficient2D>> vs_coefs;
    std::map<int, std::unique_ptr<InterpolatingCoefficient2D>> rho_coefs;
    std::map<int, std::unique_ptr<InterpolatingCoefficient2D>> qkappa_coefs;
    std::map<int, std::unique_ptr<InterpolatingCoefficient2D>> qmu_coefs;

    for (const auto& entry : entries) {
        if (entry.is_heterogeneous && entry.grid_data_2d.has_value()) {
            const auto& g = *entry.grid_data_2d;
            vp_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient2D>(
                g.vp, g.nx, g.ny, g.dx, g.dy, g.x0, g.y0);
            vs_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient2D>(
                g.vs, g.nx, g.ny, g.dx, g.dy, g.x0, g.y0);
            rho_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient2D>(
                g.rho, g.nx, g.ny, g.dx, g.dy, g.x0, g.y0);
            if (attenuation.enabled && g.has_Q) {
                qkappa_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient2D>(
                    g.Qkappa, g.nx, g.ny, g.dx, g.dy, g.x0, g.y0);
                qmu_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient2D>(
                    g.Qmu, g.nx, g.ny, g.dx, g.dy, g.x0, g.y0);
            }
        }
    }

    // Create material
    auto material = std::make_unique<IsotropicElasticMaterial>(ne, ngll, ngll);

    // Allocate Q fields if attenuation is enabled
    if (attenuation.enabled) {
        material->AllocateAttenuationFields(attenuation.f0, attenuation.n_units);
    }

    auto lambda_view = material->Lambda().ViewHostWrite();
    auto mu_view = material->Mu().ViewHostWrite();
    auto rho_view = material->Rho().ViewHostWrite();

    Mesh* mesh = fes.GetMesh();

    // Fill each element based on its attribute
    for (int e = 0; e < ne; e++) {
        int attr = mesh->GetAttribute(e);
        auto it = attr_map.find(attr);
        MFEM_VERIFY(it != attr_map.end(),
            "No material defined for attribute " << attr);

        const auto* entry = it->second;

        if (entry->mode == "adios2") {
            // ADIOS2 sub-mode: pre-computed GLL data, layout
            // [ne * ngll * ngll]. Slice this element's chunk.
            MFEM_VERIFY(entry->adios2_vp && entry->adios2_vs && entry->adios2_rho,
                "Attribute " << attr
                << " adios2 entry missing vp/vs/rho field");
            const real_t* vp_data  = entry->adios2_vp->Data().HostRead();
            const real_t* vs_data  = entry->adios2_vs->Data().HostRead();
            const real_t* rho_data = entry->adios2_rho->Data().HostRead();
            int elem_offset = e * ngll * ngll;
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    int ip = j * ngll + i;
                    real_t vp = vp_data[elem_offset + ip];
                    real_t vs = vs_data[elem_offset + ip];
                    real_t rho_val = rho_data[elem_offset + ip];
                    auto [lambda, mu] = VelocityToLame(vp, vs, rho_val);
                    lambda_view(i, j, e) = lambda;
                    mu_view(i, j, e) = mu;
                    rho_view(i, j, e) = rho_val;
                }
            }
            if (attenuation.enabled
                && entry->adios2_qkappa && entry->adios2_qmu) {
                auto qkappa_view = material->Qkappa().ViewHostWrite();
                auto qmu_view    = material->Qmu().ViewHostWrite();
                const real_t* qk = entry->adios2_qkappa->Data().HostRead();
                const real_t* qm = entry->adios2_qmu->Data().HostRead();
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        int ip = j * ngll + i;
                        qkappa_view(i, j, e) = qk[elem_offset + ip];
                        qmu_view(i, j, e)    = qm[elem_offset + ip];
                    }
                }
            }
        }
        else if (entry->is_heterogeneous) {
            // Grid interpolation
            ElementTransformation* Tr = fes.GetElementTransformation(e);
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    int ip_idx = j * ngll + i;
                    const IntegrationPoint& ip = ir.IntPoint(ip_idx);
                    Tr->SetIntPoint(&ip);

                    real_t vp = vp_coefs.at(attr)->Eval(*Tr, ip);
                    real_t vs = vs_coefs.at(attr)->Eval(*Tr, ip);
                    real_t rho_val = rho_coefs.at(attr)->Eval(*Tr, ip);

                    auto [lambda, mu] = VelocityToLame(vp, vs, rho_val);
                    lambda_view(i, j, e) = lambda;
                    mu_view(i, j, e) = mu;
                    rho_view(i, j, e) = rho_val;
                }
            }

            // Set Q values from grid if available
            if (attenuation.enabled && qkappa_coefs.count(attr) > 0) {
                auto qkappa_view = material->Qkappa().ViewHostWrite();
                auto qmu_view = material->Qmu().ViewHostWrite();
                ElementTransformation* Tr = fes.GetElementTransformation(e);
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        int ip_idx = j * ngll + i;
                        const IntegrationPoint& ip = ir.IntPoint(ip_idx);
                        Tr->SetIntPoint(&ip);

                        qkappa_view(i, j, e) = qkappa_coefs.at(attr)->Eval(*Tr, ip);
                        qmu_view(i, j, e) = qmu_coefs.at(attr)->Eval(*Tr, ip);
                    }
                }
            }
        } else {
            // Constant values
            auto [lambda, mu] = VelocityToLame(entry->vp, entry->vs, entry->rho);
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    lambda_view(i, j, e) = lambda;
                    mu_view(i, j, e) = mu;
                    rho_view(i, j, e) = entry->rho;
                }
            }

            // Set Q values if attenuation enabled and values are valid
            if (attenuation.enabled && entry->Qkappa > 0 && entry->Qmu > 0) {
                auto qkappa_view = material->Qkappa().ViewHostWrite();
                auto qmu_view = material->Qmu().ViewHostWrite();
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        qkappa_view(i, j, e) = entry->Qkappa;
                        qmu_view(i, j, e) = entry->Qmu;
                    }
                }
            }
        }
    }

    material->ComputeKappaFromLambdaMu();
    return material;
}

std::unique_ptr<IsotropicElasticMaterial> IsotropicElasticBuilder<2>::BuildFromADIOS2(
    const IsotropicElasticInput& data,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    MFEM_VERIFY(data.adios2_vp && data.adios2_vs && data.adios2_rho,
        "ADIOS2 elastic data requires vp, vs, and rho");

    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    const auto& vp_field = *data.adios2_vp;
    const auto& vs_field = *data.adios2_vs;
    const auto& rho_field = *data.adios2_rho;

    MFEM_VERIFY(vp_field.NumElements() == ne,
        "ADIOS2 vp has " << vp_field.NumElements() << " elements but mesh has " << ne);

    auto material = std::make_unique<IsotropicElasticMaterial>(ne, ngll, ngll);

    auto lambda_view = material->Lambda().ViewHostWrite();
    auto mu_view     = material->Mu().ViewHostWrite();
    auto rho_view    = material->Rho().ViewHostWrite();

    const real_t* vp_data  = vp_field.Data().HostRead();
    const real_t* vs_data  = vs_field.Data().HostRead();
    const real_t* rho_data = rho_field.Data().HostRead();

    for (int e = 0; e < ne; e++) {
        int elem_offset = e * ngll * ngll;
        for (int j = 0; j < ngll; j++) {
            for (int i = 0; i < ngll; i++) {
                int ip = j * ngll + i;
                real_t vp = vp_data[elem_offset + ip];
                real_t vs = vs_data[elem_offset + ip];
                real_t rho_val = rho_data[elem_offset + ip];
                auto [lambda, mu] = VelocityToLame(vp, vs, rho_val);
                lambda_view(i, j, e) = lambda;
                mu_view(i, j, e) = mu;
                rho_view(i, j, e) = rho_val;
            }
        }
    }

    // Attenuation: copy Qkappa and Qmu directly from ADIOS2
    if (data.attenuation.enabled && data.adios2_qkappa && data.adios2_qmu) {
        material->AllocateAttenuationFields(data.attenuation.f0, data.attenuation.n_units);
        auto qkappa_view = material->Qkappa().ViewHostWrite();
        auto qmu_view    = material->Qmu().ViewHostWrite();
        const real_t* qk = data.adios2_qkappa->Data().HostRead();
        const real_t* qm = data.adios2_qmu->Data().HostRead();
        for (int e = 0; e < ne; e++) {
            int elem_offset = e * ngll * ngll;
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    int ip = j * ngll + i;
                    qkappa_view(i, j, e) = qk[elem_offset + ip];
                    qmu_view(i, j, e)    = qm[elem_offset + ip];
                }
            }
        }
    }

    material->ComputeKappaFromLambdaMu();
    return material;
}

void IsotropicElasticBuilder<2>::GetMeshInfo(
    const ParFiniteElementSpace& fes, const IntegrationRule& ir,
    int& ne, int& ngll)
{
    ne = fes.GetNE();
    int npts = ir.GetNPoints();
    ngll = (int)std::sqrt((real_t)npts);
    MFEM_VERIFY(ngll * ngll == npts, "Integration rule must be square");
}

// =============================================================================
// 3D Builder Implementation
// =============================================================================

std::unique_ptr<IsotropicElasticMaterial3D> IsotropicElasticBuilder<3>::Build(
    const IsotropicElasticInput& data,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    MFEM_VERIFY(data.IsValid(), "Invalid IsotropicElasticInput");

    std::unique_ptr<IsotropicElasticMaterial3D> material;

    switch (data.source) {
        case DataSource::Constant:
            material = BuildFromConstant(*data.constant, data.attenuation, fes, ir);
            break;
        case DataSource::Grid:
            material = BuildFromGrid(data, fes, ir);
            break;
        case DataSource::ByAttribute:
            material = BuildFromByAttribute(*data.by_attribute, data.attenuation, fes, ir);
            break;
        case DataSource::ByAttributeMixed:
            material = BuildFromByAttributeMixed(*data.by_attribute_mixed, data.attenuation, fes, ir);
            break;
        case DataSource::ADIOS2:
            material = BuildFromADIOS2(data, fes, ir);
            break;
    }

    //debug
    // auto lambda = material->Lambda().ToParGridFunction(&fes);
    // auto mu = material->Mu().ToParGridFunction(&fes);
    // auto rho = material->Rho().ToParGridFunction(&fes);

    // Compute Vp = sqrt((lambda + 2*mu) / rho)
    // ParGridFunction vp(&fes);
    // for (int i = 0; i < vp.Size(); i++) {
    //     real_t lam = (*lambda)[i];
    //     real_t m = (*mu)[i];
    //     real_t r = (*rho)[i];
    //     vp[i] = std::sqrt((lam + 2.0 * m) / r);
    // }

    // Automatically apply attenuation correction
    if (data.attenuation.enabled && material->HasAttenuation()) {
        material->ApplyAttenuationCorrection();
    }

    return material;
}

std::unique_ptr<IsotropicElasticMaterial3D> IsotropicElasticBuilder<3>::BuildFromConstant(
    const IsotropicElasticConstantParams& params,
    const MaterialAttenuationConfig& attenuation,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    auto [lambda, mu] = VelocityToLame(params.vp, params.vs, params.rho);

    auto material = std::make_unique<IsotropicElasticMaterial3D>(ne, ngll, ngll, ngll);
    material->Lambda().SetConstant(lambda);
    material->Mu().SetConstant(mu);
    material->Rho().SetConstant(params.rho);
    material->ComputeKappaFromLambdaMu();

    if (attenuation.enabled) {
        material->InitializeAttenuationConstant(
            params.qkappa, params.qmu,
            attenuation.f0, attenuation.n_units);
    }

    return material;
}

std::unique_ptr<IsotropicElasticMaterial3D> IsotropicElasticBuilder<3>::BuildFromGrid(
    const IsotropicElasticInput& data,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    MFEM_VERIFY(data.grid_3d.has_value(), "3D grid data not available");
    const auto& grid = *data.grid_3d;

    // Create interpolating coefficients
    InterpolatingCoefficient3D vp_coef(
        grid.vp, grid.nx, grid.ny, grid.nz,
        grid.dx, grid.dy, grid.dz,
        grid.x0, grid.y0, grid.z0);
    InterpolatingCoefficient3D vs_coef(
        grid.vs, grid.nx, grid.ny, grid.nz,
        grid.dx, grid.dy, grid.dz,
        grid.x0, grid.y0, grid.z0);
    InterpolatingCoefficient3D rho_coef(
        grid.rho, grid.nx, grid.ny, grid.nz,
        grid.dx, grid.dy, grid.dz,
        grid.x0, grid.y0, grid.z0);

    // Get mesh info
    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    // Create material and initialize from coefficients
    auto material = std::make_unique<IsotropicElasticMaterial3D>(ne, ngll, ngll, ngll);

    // Initialize lambda, mu, rho from velocity coefficients
    real_t* lambda_data = material->Lambda().GetData();
    real_t* mu_data = material->Mu().GetData();
    real_t* rho_data = material->Rho().GetData();

    int stride_e = ngll * ngll * ngll;

    for (int e = 0; e < ne; e++) {
        ElementTransformation* Tr = fes.GetElementTransformation(e);

        for (int k = 0; k < ngll; k++) {
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    int ip_idx = k * ngll * ngll + j * ngll + i;
                    const IntegrationPoint& ip = ir.IntPoint(ip_idx);
                    Tr->SetIntPoint(&ip);

                    real_t vp = vp_coef.Eval(*Tr, ip);
                    real_t vs = vs_coef.Eval(*Tr, ip);
                    real_t rho = rho_coef.Eval(*Tr, ip);

                    auto [lambda, mu] = VelocityToLame(vp, vs, rho);

                    int flat_idx = e * stride_e + k * ngll * ngll + j * ngll + i;
                    lambda_data[flat_idx] = lambda;
                    mu_data[flat_idx] = mu;
                    rho_data[flat_idx] = rho;
                }
            }
        }
    }

    material->ComputeKappaFromLambdaMu();

    // Initialize attenuation from grid data
    if (data.attenuation.enabled && grid.has_Q) {
        InterpolatingCoefficient3D qkappa_coef(
            grid.Qkappa, grid.nx, grid.ny, grid.nz,
            grid.dx, grid.dy, grid.dz,
            grid.x0, grid.y0, grid.z0);
        InterpolatingCoefficient3D qmu_coef(
            grid.Qmu, grid.nx, grid.ny, grid.nz,
            grid.dx, grid.dy, grid.dz,
            grid.x0, grid.y0, grid.z0);
        material->InitializeAttenuationFromCoefficient(
            qkappa_coef, qmu_coef, fes, ir,
            data.attenuation.f0, data.attenuation.n_units);
    }

    return material;
}

std::unique_ptr<IsotropicElasticMaterial3D> IsotropicElasticBuilder<3>::BuildFromByAttribute(
    const std::vector<IsotropicElasticAttributeEntry>& entries,
    const MaterialAttenuationConfig& attenuation,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    // Build attribute map
    std::map<int, IsotropicElasticAttributeEntry> attr_map;
    for (const auto& entry : entries) {
        attr_map[entry.attribute] = entry;
    }

    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    auto material = std::make_unique<IsotropicElasticMaterial3D>(ne, ngll, ngll, ngll);

    // Allocate Q fields if attenuation is enabled
    if (attenuation.enabled) {
        material->AllocateAttenuationFields(attenuation.f0, attenuation.n_units);
    }

    // Get ViewWrite for each field
    auto lambda_view = material->Lambda().ViewHostWrite();
    auto mu_view = material->Mu().ViewHostWrite();
    auto rho_view = material->Rho().ViewHostWrite();

    Mesh* mesh = fes.GetMesh();

    // Fill each element based on its attribute
    for (int e = 0; e < ne; e++) {
        int attr = mesh->GetAttribute(e);
        auto it = attr_map.find(attr);
        MFEM_VERIFY(it != attr_map.end(),
            "No material defined for attribute " << attr);

        const auto& entry = it->second;
        auto [lambda, mu] = VelocityToLame(entry.vp, entry.vs, entry.rho);

        for (int k = 0; k < ngll; k++) {
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    lambda_view(i, j, k, e) = lambda;
                    mu_view(i, j, k, e) = mu;
                    rho_view(i, j, k, e) = entry.rho;
                }
            }
        }

        // Set Q values if attenuation enabled
        if (attenuation.enabled) {
            auto qkappa_view = material->Qkappa().ViewHostWrite();
            auto qmu_view = material->Qmu().ViewHostWrite();
            for (int k = 0; k < ngll; k++) {
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        qkappa_view(i, j, k, e) = entry.Qkappa;
                        qmu_view(i, j, k, e) = entry.Qmu;
                    }
                }
            }
        }
    }

    material->ComputeKappaFromLambdaMu();
    return material;
}

std::unique_ptr<IsotropicElasticMaterial3D> IsotropicElasticBuilder<3>::BuildFromByAttributeMixed(
    const std::vector<IsotropicElasticMixedAttributeEntry>& entries,
    const MaterialAttenuationConfig& attenuation,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    // Build attribute map
    std::map<int, const IsotropicElasticMixedAttributeEntry*> attr_map;
    for (const auto& entry : entries) {
        attr_map[entry.attribute] = &entry;
    }

    // Create interpolating coefficients for heterogeneous attributes
    std::map<int, std::unique_ptr<InterpolatingCoefficient3D>> vp_coefs;
    std::map<int, std::unique_ptr<InterpolatingCoefficient3D>> vs_coefs;
    std::map<int, std::unique_ptr<InterpolatingCoefficient3D>> rho_coefs;
    std::map<int, std::unique_ptr<InterpolatingCoefficient3D>> qkappa_coefs;
    std::map<int, std::unique_ptr<InterpolatingCoefficient3D>> qmu_coefs;

    for (const auto& entry : entries) {
        if (entry.is_heterogeneous && entry.grid_data_3d.has_value()) {
            const auto& g = *entry.grid_data_3d;
            vp_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient3D>(
                g.vp, g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.x0, g.y0, g.z0);
            vs_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient3D>(
                g.vs, g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.x0, g.y0, g.z0);
            rho_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient3D>(
                g.rho, g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.x0, g.y0, g.z0);
            if (attenuation.enabled && g.has_Q) {
                qkappa_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient3D>(
                    g.Qkappa, g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.x0, g.y0, g.z0);
                qmu_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient3D>(
                    g.Qmu, g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.x0, g.y0, g.z0);
            }
        }
    }

    // Create material
    auto material = std::make_unique<IsotropicElasticMaterial3D>(ne, ngll, ngll, ngll);

    // Allocate Q fields if attenuation is enabled
    if (attenuation.enabled) {
        material->AllocateAttenuationFields(attenuation.f0, attenuation.n_units);
    }

    auto lambda_view = material->Lambda().ViewHostWrite();
    auto mu_view = material->Mu().ViewHostWrite();
    auto rho_view = material->Rho().ViewHostWrite();

    Mesh* mesh = fes.GetMesh();

    // Fill each element based on its attribute
    for (int e = 0; e < ne; e++) {
        int attr = mesh->GetAttribute(e);
        auto it = attr_map.find(attr);
        MFEM_VERIFY(it != attr_map.end(),
            "No material defined for attribute " << attr);

        const auto* entry = it->second;

        if (entry->mode == "adios2") {
            // ADIOS2 sub-mode: pre-computed GLL data, layout
            // [ne * ngll * ngll * ngll]. Slice this element's chunk.
            MFEM_VERIFY(entry->adios2_vp && entry->adios2_vs && entry->adios2_rho,
                "Attribute " << attr
                << " adios2 entry missing vp/vs/rho field");
            const real_t* vp_data  = entry->adios2_vp->Data().HostRead();
            const real_t* vs_data  = entry->adios2_vs->Data().HostRead();
            const real_t* rho_data = entry->adios2_rho->Data().HostRead();
            int elem_offset = e * ngll * ngll * ngll;
            for (int k = 0; k < ngll; k++) {
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        int ip = k * ngll * ngll + j * ngll + i;
                        real_t vp = vp_data[elem_offset + ip];
                        real_t vs = vs_data[elem_offset + ip];
                        real_t rho_val = rho_data[elem_offset + ip];
                        auto [lambda, mu] = VelocityToLame(vp, vs, rho_val);
                        lambda_view(i, j, k, e) = lambda;
                        mu_view(i, j, k, e) = mu;
                        rho_view(i, j, k, e) = rho_val;
                    }
                }
            }
            if (attenuation.enabled
                && entry->adios2_qkappa && entry->adios2_qmu) {
                auto qkappa_view = material->Qkappa().ViewHostWrite();
                auto qmu_view    = material->Qmu().ViewHostWrite();
                const real_t* qk = entry->adios2_qkappa->Data().HostRead();
                const real_t* qm = entry->adios2_qmu->Data().HostRead();
                for (int k = 0; k < ngll; k++) {
                    for (int j = 0; j < ngll; j++) {
                        for (int i = 0; i < ngll; i++) {
                            int ip = k * ngll * ngll + j * ngll + i;
                            qkappa_view(i, j, k, e) = qk[elem_offset + ip];
                            qmu_view(i, j, k, e)    = qm[elem_offset + ip];
                        }
                    }
                }
            }
        }
        else if (entry->is_heterogeneous) {
            // Grid interpolation
            ElementTransformation* Tr = fes.GetElementTransformation(e);
            for (int k = 0; k < ngll; k++) {
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        int ip_idx = k * ngll * ngll + j * ngll + i;
                        const IntegrationPoint& ip = ir.IntPoint(ip_idx);
                        Tr->SetIntPoint(&ip);

                        real_t vp = vp_coefs.at(attr)->Eval(*Tr, ip);
                        real_t vs = vs_coefs.at(attr)->Eval(*Tr, ip);
                        real_t rho_val = rho_coefs.at(attr)->Eval(*Tr, ip);

                        auto [lambda, mu] = VelocityToLame(vp, vs, rho_val);
                        lambda_view(i, j, k, e) = lambda;
                        mu_view(i, j, k, e) = mu;
                        rho_view(i, j, k, e) = rho_val;
                    }
                }
            }

            // Set Q values from grid if available
            if (attenuation.enabled && qkappa_coefs.count(attr) > 0) {
                auto qkappa_view = material->Qkappa().ViewHostWrite();
                auto qmu_view = material->Qmu().ViewHostWrite();
                ElementTransformation* Tr = fes.GetElementTransformation(e);
                for (int k = 0; k < ngll; k++) {
                    for (int j = 0; j < ngll; j++) {
                        for (int i = 0; i < ngll; i++) {
                            int ip_idx = k * ngll * ngll + j * ngll + i;
                            const IntegrationPoint& ip = ir.IntPoint(ip_idx);
                            Tr->SetIntPoint(&ip);

                            qkappa_view(i, j, k, e) = qkappa_coefs.at(attr)->Eval(*Tr, ip);
                            qmu_view(i, j, k, e) = qmu_coefs.at(attr)->Eval(*Tr, ip);
                        }
                    }
                }
            }
        } else {
            // Constant values
            auto [lambda, mu] = VelocityToLame(entry->vp, entry->vs, entry->rho);
            for (int k = 0; k < ngll; k++) {
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        lambda_view(i, j, k, e) = lambda;
                        mu_view(i, j, k, e) = mu;
                        rho_view(i, j, k, e) = entry->rho;
                    }
                }
            }

            // Set Q values if attenuation enabled and values are valid
            if (attenuation.enabled && entry->Qkappa > 0 && entry->Qmu > 0) {
                auto qkappa_view = material->Qkappa().ViewHostWrite();
                auto qmu_view = material->Qmu().ViewHostWrite();
                for (int k = 0; k < ngll; k++) {
                    for (int j = 0; j < ngll; j++) {
                        for (int i = 0; i < ngll; i++) {
                            qkappa_view(i, j, k, e) = entry->Qkappa;
                            qmu_view(i, j, k, e) = entry->Qmu;
                        }
                    }
                }
            }
        }
    }

    material->ComputeKappaFromLambdaMu();
    return material;
}

std::unique_ptr<IsotropicElasticMaterial3D> IsotropicElasticBuilder<3>::BuildFromADIOS2(
    const IsotropicElasticInput& data,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    MFEM_VERIFY(data.adios2_vp && data.adios2_vs && data.adios2_rho,
        "ADIOS2 elastic data requires vp, vs, and rho");

    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    const auto& vp_field = *data.adios2_vp;
    const auto& vs_field = *data.adios2_vs;
    const auto& rho_field = *data.adios2_rho;

    MFEM_VERIFY(vp_field.NumElements() == ne,
        "ADIOS2 vp has " << vp_field.NumElements() << " elements but mesh has " << ne);

    auto material = std::make_unique<IsotropicElasticMaterial3D>(ne, ngll, ngll, ngll);

    auto lambda_view = material->Lambda().ViewHostWrite();
    auto mu_view     = material->Mu().ViewHostWrite();
    auto rho_view    = material->Rho().ViewHostWrite();

    const real_t* vp_data  = vp_field.Data().HostRead();
    const real_t* vs_data  = vs_field.Data().HostRead();
    const real_t* rho_data = rho_field.Data().HostRead();

    for (int e = 0; e < ne; e++) {
        int elem_offset = e * ngll * ngll * ngll;
        for (int k = 0; k < ngll; k++) {
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    int ip = k * ngll * ngll + j * ngll + i;
                    real_t vp = vp_data[elem_offset + ip];
                    real_t vs = vs_data[elem_offset + ip];
                    real_t rho_val = rho_data[elem_offset + ip];
                    auto [lambda, mu] = VelocityToLame(vp, vs, rho_val);
                    lambda_view(i, j, k, e) = lambda;
                    mu_view(i, j, k, e) = mu;
                    rho_view(i, j, k, e) = rho_val;
                }
            }
        }
    }

    // Attenuation: copy Qkappa and Qmu directly from ADIOS2
    if (data.attenuation.enabled && data.adios2_qkappa && data.adios2_qmu) {
        material->AllocateAttenuationFields(data.attenuation.f0, data.attenuation.n_units);
        auto qkappa_view = material->Qkappa().ViewHostWrite();
        auto qmu_view    = material->Qmu().ViewHostWrite();
        const real_t* qk = data.adios2_qkappa->Data().HostRead();
        const real_t* qm = data.adios2_qmu->Data().HostRead();
        for (int e = 0; e < ne; e++) {
            int elem_offset = e * ngll * ngll * ngll;
            for (int k = 0; k < ngll; k++) {
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        int ip = k * ngll * ngll + j * ngll + i;
                        qkappa_view(i, j, k, e) = qk[elem_offset + ip];
                        qmu_view(i, j, k, e)    = qm[elem_offset + ip];
                    }
                }
            }
        }
    }

    material->ComputeKappaFromLambdaMu();
    return material;
}

void IsotropicElasticBuilder<3>::GetMeshInfo(
    const ParFiniteElementSpace& fes, const IntegrationRule& ir,
    int& ne, int& ngll)
{
    ne = fes.GetNE();
    int npts = ir.GetNPoints();
    ngll = (int)std::cbrt((real_t)npts);
    MFEM_VERIFY(ngll * ngll * ngll == npts, "Integration rule must be cubic");
}

}  // namespace SEM
