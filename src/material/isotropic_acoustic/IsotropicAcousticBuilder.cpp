/**
 * @file IsotropicAcousticBuilder.cpp
 * @brief Implementation of IsotropicAcousticBuilder
 */

#include "material/isotropic_acoustic/IsotropicAcousticBuilder.hpp"
#include "material/MaterialUtils.hpp"
#include <mfem.hpp>
#include <map>
#include <cmath>

namespace SEM {

// =============================================================================
// 2D Builder Implementation
// =============================================================================

std::unique_ptr<IsotropicAcousticMaterial> IsotropicAcousticBuilder<2>::Build(
    const IsotropicAcousticInput& data,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    MFEM_VERIFY(data.IsValid(), "Invalid IsotropicAcousticInput");

    std::unique_ptr<IsotropicAcousticMaterial> material;

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

    // //save for debug
    // auto kappa_gf = material->Kappa().ToParGridFunction(&fes);
    // kappa_gf->Save("kappa.gf");

    // // Save Vp for debug: Vp = sqrt(kappa * inv_rho)
    // auto invrho_gf = material->InvRho().ToParGridFunction(&fes);
    // ParGridFunction vp_gf(&fes);
    // for (int i = 0; i < vp_gf.Size(); i++) {
    //     vp_gf(i) = std::sqrt((*kappa_gf)(i) * (*invrho_gf)(i));
    // }
    // vp_gf.Save("vp.gf");

    // Automatically apply attenuation correction
    if (data.attenuation.enabled && material->HasAttenuation()) {
        material->ApplyAttenuationCorrection();
    }

    return material;
}

std::unique_ptr<IsotropicAcousticMaterial> IsotropicAcousticBuilder<2>::BuildFromConstant(
    const IsotropicAcousticConstantParams& params,
    const MaterialAttenuationConfig& attenuation,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    real_t kappa = VelocityToKappa(params.vp, params.rho);
    real_t inv_rho = 1.0 / params.rho;

    auto material = std::make_unique<IsotropicAcousticMaterial>(ne, ngll, ngll);
    material->Kappa().SetConstant(kappa);
    material->InvRho().SetConstant(inv_rho);

    if (attenuation.enabled) {
        material->InitializeAttenuationConstant(
            params.qkappa, attenuation.f0, attenuation.n_units);
    }

    return material;
}

std::unique_ptr<IsotropicAcousticMaterial> IsotropicAcousticBuilder<2>::BuildFromGrid(
    const IsotropicAcousticInput& data,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    MFEM_VERIFY(data.grid_2d.has_value(), "2D grid data not available");
    const auto& grid = *data.grid_2d;

    // Create interpolating coefficients
    InterpolatingCoefficient2D vp_coef(
        grid.vp, grid.nx, grid.ny, grid.dx, grid.dy, grid.x0, grid.y0);
    InterpolatingCoefficient2D rho_coef(
        grid.rho, grid.nx, grid.ny, grid.dx, grid.dy, grid.x0, grid.y0);

    auto material = std::make_unique<IsotropicAcousticMaterial>(
        vp_coef, rho_coef, fes, ir);

    // Initialize attenuation from grid data
    if (data.attenuation.enabled && grid.has_Q) {
        InterpolatingCoefficient2D qkappa_coef(
            grid.Qkappa, grid.nx, grid.ny, grid.dx, grid.dy, grid.x0, grid.y0);
        material->InitializeAttenuationFromCoefficient(
            qkappa_coef, fes, ir,
            data.attenuation.f0, data.attenuation.n_units);
    }

    return material;
}

std::unique_ptr<IsotropicAcousticMaterial> IsotropicAcousticBuilder<2>::BuildFromByAttribute(
    const std::vector<IsotropicAcousticAttributeEntry>& entries,
    const MaterialAttenuationConfig& attenuation,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    // Build attribute map
    std::map<int, IsotropicAcousticAttributeEntry> attr_map;
    for (const auto& entry : entries) {
        attr_map[entry.attribute] = entry;
    }

    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    auto material = std::make_unique<IsotropicAcousticMaterial>(ne, ngll, ngll);

    // Allocate Q field if attenuation is enabled
    if (attenuation.enabled) {
        material->AllocateAttenuationFields(attenuation.f0, attenuation.n_units);
    }

    // Get ViewWrite for each field
    auto kappa_view = material->Kappa().ViewHostWrite();
    auto invrho_view = material->InvRho().ViewHostWrite();

    Mesh* mesh = fes.GetMesh();

    // Fill each element based on its attribute
    for (int e = 0; e < ne; e++) {
        int attr = mesh->GetAttribute(e);
        auto it = attr_map.find(attr);
        MFEM_VERIFY(it != attr_map.end(),
            "No acoustic material defined for attribute " << attr);

        const auto& entry = it->second;
        real_t kappa = VelocityToKappa(entry.vp, entry.rho);
        real_t inv_rho = 1.0 / entry.rho;

        for (int j = 0; j < ngll; j++) {
            for (int i = 0; i < ngll; i++) {
                kappa_view(i, j, e) = kappa;
                invrho_view(i, j, e) = inv_rho;
            }
        }

        // Set Q values if attenuation enabled
        if (attenuation.enabled) {
            auto qkappa_view = material->Qkappa().ViewHostWrite();
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    qkappa_view(i, j, e) = entry.Qkappa;
                }
            }
        }
    }

    return material;
}

std::unique_ptr<IsotropicAcousticMaterial> IsotropicAcousticBuilder<2>::BuildFromByAttributeMixed(
    const std::vector<IsotropicAcousticMixedAttributeEntry>& entries,
    const MaterialAttenuationConfig& attenuation,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    // Build attribute map
    std::map<int, const IsotropicAcousticMixedAttributeEntry*> attr_map;
    for (const auto& entry : entries) {
        attr_map[entry.attribute] = &entry;
    }

    // Create interpolating coefficients for heterogeneous attributes
    std::map<int, std::unique_ptr<InterpolatingCoefficient2D>> vp_coefs;
    std::map<int, std::unique_ptr<InterpolatingCoefficient2D>> rho_coefs;
    std::map<int, std::unique_ptr<InterpolatingCoefficient2D>> qkappa_coefs;

    for (const auto& entry : entries) {
        if (entry.is_heterogeneous && entry.grid_data_2d.has_value()) {
            const auto& g = *entry.grid_data_2d;
            vp_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient2D>(
                g.vp, g.nx, g.ny, g.dx, g.dy, g.x0, g.y0);
            rho_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient2D>(
                g.rho, g.nx, g.ny, g.dx, g.dy, g.x0, g.y0);
            if (attenuation.enabled && g.has_Q) {
                qkappa_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient2D>(
                    g.Qkappa, g.nx, g.ny, g.dx, g.dy, g.x0, g.y0);
            }
        }
    }

    // Create material
    auto material = std::make_unique<IsotropicAcousticMaterial>(ne, ngll, ngll);

    // Allocate Q field if attenuation is enabled
    if (attenuation.enabled) {
        material->AllocateAttenuationFields(attenuation.f0, attenuation.n_units);
    }

    auto kappa_view = material->Kappa().ViewHostWrite();
    auto invrho_view = material->InvRho().ViewHostWrite();

    Mesh* mesh = fes.GetMesh();

    // Fill each element based on its attribute
    for (int e = 0; e < ne; e++) {
        int attr = mesh->GetAttribute(e);
        auto it = attr_map.find(attr);
        MFEM_VERIFY(it != attr_map.end(),
            "No acoustic material defined for attribute " << attr);

        const auto* entry = it->second;

        if (entry->mode == "adios2") {
            // ADIOS2 sub-mode: pre-computed GLL data laid out as
            // [ne * ngll * ngll]. Slice out this element's chunk.
            MFEM_VERIFY(entry->adios2_vp && entry->adios2_rho,
                "Attribute " << attr << " adios2 entry missing vp/rho field");
            const real_t* vp_data  = entry->adios2_vp->Data().HostRead();
            const real_t* rho_data = entry->adios2_rho->Data().HostRead();
            int elem_offset = e * ngll * ngll;
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    int ip = j * ngll + i;
                    real_t vp = vp_data[elem_offset + ip];
                    real_t rho_val = rho_data[elem_offset + ip];
                    kappa_view(i, j, e) = VelocityToKappa(vp, rho_val);
                    invrho_view(i, j, e) = 1.0 / rho_val;
                }
            }
            if (attenuation.enabled && entry->adios2_qkappa) {
                auto qkappa_view = material->Qkappa().ViewHostWrite();
                const real_t* q_data = entry->adios2_qkappa->Data().HostRead();
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        int ip = j * ngll + i;
                        qkappa_view(i, j, e) = q_data[elem_offset + ip];
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
                    real_t rho_val = rho_coefs.at(attr)->Eval(*Tr, ip);

                    real_t kappa = VelocityToKappa(vp, rho_val);
                    real_t inv_rho = 1.0 / rho_val;
                    kappa_view(i, j, e) = kappa;
                    invrho_view(i, j, e) = inv_rho;
                }
            }

            // Set Q values from grid if available
            if (attenuation.enabled && qkappa_coefs.count(attr) > 0) {
                auto qkappa_view = material->Qkappa().ViewHostWrite();
                ElementTransformation* Tr = fes.GetElementTransformation(e);
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        int ip_idx = j * ngll + i;
                        const IntegrationPoint& ip = ir.IntPoint(ip_idx);
                        Tr->SetIntPoint(&ip);

                        qkappa_view(i, j, e) = qkappa_coefs.at(attr)->Eval(*Tr, ip);
                    }
                }
            }
        } else {
            // Constant values
            real_t kappa = VelocityToKappa(entry->vp, entry->rho);
            real_t inv_rho = 1.0 / entry->rho;
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    kappa_view(i, j, e) = kappa;
                    invrho_view(i, j, e) = inv_rho;
                }
            }

            // Set Q value if attenuation enabled and value is valid
            if (attenuation.enabled && entry->Qkappa > 0) {
                auto qkappa_view = material->Qkappa().ViewHostWrite();
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        qkappa_view(i, j, e) = entry->Qkappa;
                    }
                }
            }
        }
    }

    return material;
}

std::unique_ptr<IsotropicAcousticMaterial> IsotropicAcousticBuilder<2>::BuildFromADIOS2(
    const IsotropicAcousticInput& data,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    MFEM_VERIFY(data.adios2_vp && data.adios2_rho, "ADIOS2 data requires vp and rho");

    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    const auto& vp_field = *data.adios2_vp;
    const auto& rho_field = *data.adios2_rho;

    MFEM_VERIFY(vp_field.NumElements() == ne,
        "ADIOS2 vp has " << vp_field.NumElements() << " elements but mesh has " << ne);

    auto material = std::make_unique<IsotropicAcousticMaterial>(ne, ngll, ngll);

    // Convert vp, rho → kappa, inv_rho
    const real_t* vp_data = vp_field.Data().HostRead();
    const real_t* rho_data = rho_field.Data().HostRead();
    real_t* kappa_out = material->Kappa().Data().HostWrite();
    real_t* invrho_out = material->InvRho().Data().HostWrite();

    int total = ne * ngll * ngll;
    for (int i = 0; i < total; i++) {
        kappa_out[i] = rho_data[i] * vp_data[i] * vp_data[i];
        invrho_out[i] = 1.0 / rho_data[i];
    }

    // Attenuation: copy Qkappa directly from ADIOS2
    if (data.attenuation.enabled && data.adios2_qkappa) {
        material->AllocateAttenuationFields(data.attenuation.f0, data.attenuation.n_units);
        const real_t* q_data = data.adios2_qkappa->Data().HostRead();
        real_t* q_out = material->Qkappa().Data().HostWrite();
        for (int i = 0; i < total; i++) {
            q_out[i] = q_data[i];
        }
    }

    return material;
}

void IsotropicAcousticBuilder<2>::GetMeshInfo(
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

std::unique_ptr<IsotropicAcousticMaterial3D> IsotropicAcousticBuilder<3>::Build(
    const IsotropicAcousticInput& data,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    MFEM_VERIFY(data.IsValid(), "Invalid IsotropicAcousticInput");

    std::unique_ptr<IsotropicAcousticMaterial3D> material;

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

std::unique_ptr<IsotropicAcousticMaterial3D> IsotropicAcousticBuilder<3>::BuildFromConstant(
    const IsotropicAcousticConstantParams& params,
    const MaterialAttenuationConfig& attenuation,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    real_t kappa = VelocityToKappa(params.vp, params.rho);
    real_t inv_rho = 1.0 / params.rho;

    auto material = std::make_unique<IsotropicAcousticMaterial3D>(ne, ngll, ngll, ngll);
    material->Kappa().SetConstant(kappa);
    material->InvRho().SetConstant(inv_rho);

    if (attenuation.enabled) {
        material->InitializeAttenuationConstant(
            params.qkappa, attenuation.f0, attenuation.n_units);
    }

    return material;
}

std::unique_ptr<IsotropicAcousticMaterial3D> IsotropicAcousticBuilder<3>::BuildFromGrid(
    const IsotropicAcousticInput& data,
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
    InterpolatingCoefficient3D rho_coef(
        grid.rho, grid.nx, grid.ny, grid.nz,
        grid.dx, grid.dy, grid.dz,
        grid.x0, grid.y0, grid.z0);

    auto material = std::make_unique<IsotropicAcousticMaterial3D>(
        vp_coef, rho_coef, fes, ir);

    // Initialize attenuation from grid data
    if (data.attenuation.enabled && grid.has_Q) {
        InterpolatingCoefficient3D qkappa_coef(
            grid.Qkappa, grid.nx, grid.ny, grid.nz,
            grid.dx, grid.dy, grid.dz,
            grid.x0, grid.y0, grid.z0);
        material->InitializeAttenuationFromCoefficient(
            qkappa_coef, fes, ir,
            data.attenuation.f0, data.attenuation.n_units);
    }

    return material;
}

std::unique_ptr<IsotropicAcousticMaterial3D> IsotropicAcousticBuilder<3>::BuildFromByAttribute(
    const std::vector<IsotropicAcousticAttributeEntry>& entries,
    const MaterialAttenuationConfig& attenuation,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    // Build attribute map
    std::map<int, IsotropicAcousticAttributeEntry> attr_map;
    for (const auto& entry : entries) {
        attr_map[entry.attribute] = entry;
    }

    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    auto material = std::make_unique<IsotropicAcousticMaterial3D>(ne, ngll, ngll, ngll);

    // Allocate Q field if attenuation is enabled
    if (attenuation.enabled) {
        material->AllocateAttenuationFields(attenuation.f0, attenuation.n_units);
    }

    // Get ViewWrite for each field
    auto kappa_view = material->Kappa().ViewHostWrite();
    auto invrho_view = material->InvRho().ViewHostWrite();

    Mesh* mesh = fes.GetMesh();

    // Fill each element based on its attribute
    for (int e = 0; e < ne; e++) {
        int attr = mesh->GetAttribute(e);
        auto it = attr_map.find(attr);
        MFEM_VERIFY(it != attr_map.end(),
            "No acoustic material defined for attribute " << attr);

        const auto& entry = it->second;
        real_t kappa = VelocityToKappa(entry.vp, entry.rho);
        real_t inv_rho = 1.0 / entry.rho;

        for (int k = 0; k < ngll; k++) {
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    kappa_view(i, j, k, e) = kappa;
                    invrho_view(i, j, k, e) = inv_rho;
                }
            }
        }

        // Set Q values if attenuation enabled
        if (attenuation.enabled) {
            auto qkappa_view = material->Qkappa().ViewHostWrite();
            for (int k = 0; k < ngll; k++) {
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        qkappa_view(i, j, k, e) = entry.Qkappa;
                    }
                }
            }
        }
    }

    return material;
}

std::unique_ptr<IsotropicAcousticMaterial3D> IsotropicAcousticBuilder<3>::BuildFromByAttributeMixed(
    const std::vector<IsotropicAcousticMixedAttributeEntry>& entries,
    const MaterialAttenuationConfig& attenuation,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    // Build attribute map
    std::map<int, const IsotropicAcousticMixedAttributeEntry*> attr_map;
    for (const auto& entry : entries) {
        attr_map[entry.attribute] = &entry;
    }

    // Create interpolating coefficients for heterogeneous attributes
    std::map<int, std::unique_ptr<InterpolatingCoefficient3D>> vp_coefs;
    std::map<int, std::unique_ptr<InterpolatingCoefficient3D>> rho_coefs;
    std::map<int, std::unique_ptr<InterpolatingCoefficient3D>> qkappa_coefs;

    for (const auto& entry : entries) {
        if (entry.is_heterogeneous && entry.grid_data_3d.has_value()) {
            const auto& g = *entry.grid_data_3d;
            vp_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient3D>(
                g.vp, g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.x0, g.y0, g.z0);
            rho_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient3D>(
                g.rho, g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.x0, g.y0, g.z0);
            if (attenuation.enabled && g.has_Q) {
                qkappa_coefs[entry.attribute] = std::make_unique<InterpolatingCoefficient3D>(
                    g.Qkappa, g.nx, g.ny, g.nz, g.dx, g.dy, g.dz, g.x0, g.y0, g.z0);
            }
        }
    }

    // Create material
    auto material = std::make_unique<IsotropicAcousticMaterial3D>(ne, ngll, ngll, ngll);

    // Allocate Q field if attenuation is enabled
    if (attenuation.enabled) {
        material->AllocateAttenuationFields(attenuation.f0, attenuation.n_units);
    }

    auto kappa_view = material->Kappa().ViewHostWrite();
    auto invrho_view = material->InvRho().ViewHostWrite();

    Mesh* mesh = fes.GetMesh();

    // Fill each element based on its attribute
    for (int e = 0; e < ne; e++) {
        int attr = mesh->GetAttribute(e);
        auto it = attr_map.find(attr);
        MFEM_VERIFY(it != attr_map.end(),
            "No acoustic material defined for attribute " << attr);

        const auto* entry = it->second;

        if (entry->mode == "adios2") {
            // ADIOS2 sub-mode: pre-computed GLL data, layout
            // [ne * ngll * ngll * ngll]. Slice this element's chunk.
            MFEM_VERIFY(entry->adios2_vp && entry->adios2_rho,
                "Attribute " << attr << " adios2 entry missing vp/rho field");
            const real_t* vp_data  = entry->adios2_vp->Data().HostRead();
            const real_t* rho_data = entry->adios2_rho->Data().HostRead();
            int elem_offset = e * ngll * ngll * ngll;
            for (int k = 0; k < ngll; k++) {
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        int ip = k * ngll * ngll + j * ngll + i;
                        real_t vp = vp_data[elem_offset + ip];
                        real_t rho_val = rho_data[elem_offset + ip];
                        kappa_view(i, j, k, e) = VelocityToKappa(vp, rho_val);
                        invrho_view(i, j, k, e) = 1.0 / rho_val;
                    }
                }
            }
            if (attenuation.enabled && entry->adios2_qkappa) {
                auto qkappa_view = material->Qkappa().ViewHostWrite();
                const real_t* q_data = entry->adios2_qkappa->Data().HostRead();
                for (int k = 0; k < ngll; k++) {
                    for (int j = 0; j < ngll; j++) {
                        for (int i = 0; i < ngll; i++) {
                            int ip = k * ngll * ngll + j * ngll + i;
                            qkappa_view(i, j, k, e) = q_data[elem_offset + ip];
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
                        real_t rho_val = rho_coefs.at(attr)->Eval(*Tr, ip);

                        real_t kappa = VelocityToKappa(vp, rho_val);
                        real_t inv_rho = 1.0 / rho_val;
                        kappa_view(i, j, k, e) = kappa;
                        invrho_view(i, j, k, e) = inv_rho;
                    }
                }
            }

            // Set Q values from grid if available
            if (attenuation.enabled && qkappa_coefs.count(attr) > 0) {
                auto qkappa_view = material->Qkappa().ViewHostWrite();
                ElementTransformation* Tr = fes.GetElementTransformation(e);
                for (int k = 0; k < ngll; k++) {
                    for (int j = 0; j < ngll; j++) {
                        for (int i = 0; i < ngll; i++) {
                            int ip_idx = k * ngll * ngll + j * ngll + i;
                            const IntegrationPoint& ip = ir.IntPoint(ip_idx);
                            Tr->SetIntPoint(&ip);

                            qkappa_view(i, j, k, e) = qkappa_coefs.at(attr)->Eval(*Tr, ip);
                        }
                    }
                }
            }
        } else {
            // Constant values
            real_t kappa = VelocityToKappa(entry->vp, entry->rho);
            real_t inv_rho = 1.0 / entry->rho;
            for (int k = 0; k < ngll; k++) {
                for (int j = 0; j < ngll; j++) {
                    for (int i = 0; i < ngll; i++) {
                        kappa_view(i, j, k, e) = kappa;
                        invrho_view(i, j, k, e) = inv_rho;
                    }
                }
            }

            // Set Q value if attenuation enabled and value is valid
            if (attenuation.enabled && entry->Qkappa > 0) {
                auto qkappa_view = material->Qkappa().ViewHostWrite();
                for (int k = 0; k < ngll; k++) {
                    for (int j = 0; j < ngll; j++) {
                        for (int i = 0; i < ngll; i++) {
                            qkappa_view(i, j, k, e) = entry->Qkappa;
                        }
                    }
                }
            }
        }
    }

    return material;
}

std::unique_ptr<IsotropicAcousticMaterial3D> IsotropicAcousticBuilder<3>::BuildFromADIOS2(
    const IsotropicAcousticInput& data,
    ParFiniteElementSpace& fes,
    const IntegrationRule& ir)
{
    MFEM_VERIFY(data.adios2_vp && data.adios2_rho, "ADIOS2 data requires vp and rho");

    int ne, ngll;
    GetMeshInfo(fes, ir, ne, ngll);

    const auto& vp_field = *data.adios2_vp;
    const auto& rho_field = *data.adios2_rho;

    MFEM_VERIFY(vp_field.NumElements() == ne,
        "ADIOS2 vp has " << vp_field.NumElements() << " elements but mesh has " << ne);

    auto material = std::make_unique<IsotropicAcousticMaterial3D>(ne, ngll, ngll, ngll);

    // Convert vp, rho → kappa, inv_rho
    const real_t* vp_data = vp_field.Data().HostRead();
    const real_t* rho_data = rho_field.Data().HostRead();
    real_t* kappa_out = material->Kappa().Data().HostWrite();
    real_t* invrho_out = material->InvRho().Data().HostWrite();

    int total = ne * ngll * ngll * ngll;
    for (int i = 0; i < total; i++) {
        kappa_out[i] = rho_data[i] * vp_data[i] * vp_data[i];
        invrho_out[i] = 1.0 / rho_data[i];
    }

    // Attenuation: copy Qkappa directly from ADIOS2
    if (data.attenuation.enabled && data.adios2_qkappa) {
        material->AllocateAttenuationFields(data.attenuation.f0, data.attenuation.n_units);
        const real_t* q_data = data.adios2_qkappa->Data().HostRead();
        real_t* q_out = material->Qkappa().Data().HostWrite();
        for (int i = 0; i < total; i++) {
            q_out[i] = q_data[i];
        }
    }

    return material;
}

void IsotropicAcousticBuilder<3>::GetMeshInfo(
    const ParFiniteElementSpace& fes, const IntegrationRule& ir,
    int& ne, int& ngll)
{
    ne = fes.GetNE();
    int npts = ir.GetNPoints();
    ngll = (int)std::cbrt((real_t)npts);
    MFEM_VERIFY(ngll * ngll * ngll == npts, "Integration rule must be cubic");
}

}  // namespace SEM
