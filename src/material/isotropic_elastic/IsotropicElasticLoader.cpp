/**
 * @file IsotropicElasticLoader.cpp
 * @brief Implementation of IsotropicElasticLoader
 */

#include "material/isotropic_elastic/IsotropicElasticLoader.hpp"
#include "material/isotropic_elastic/IsotropicElasticIO.hpp"
#include "io/ADIOS2IO.hpp"
#include <mfem.hpp>

namespace SEM {

// =============================================================================
// Public: Main Load function
// =============================================================================

IsotropicElasticInput IsotropicElasticLoader::Load(const MaterialConfig& config, bool is_3d) {
    IsotropicElasticInput data;
    data.is_3d = is_3d;
    data.attenuation = ConvertAttenuation(config.attenuation);

    if (config.format == "constant") {
        data = LoadConstant(config);
        data.is_3d = is_3d;  // Preserve dimension info
    }
    else if (config.format == "grid") {
        data = is_3d ? LoadGrid3D(config) : LoadGrid2D(config);
    }
    else if (config.format == "by_attribute") {
        data = LoadByAttribute(config);
        data.is_3d = is_3d;  // Preserve dimension info
    }
    else if (config.format == "by_attribute_mixed") {
        data = LoadByAttributeMixed(config, is_3d);
    }
    else if (config.format == "adios2") {
        data = LoadADIOS2(config, is_3d);
    }
    else {
        MFEM_ABORT("Unsupported material format: " + config.format +
                   " (supported: constant, grid, by_attribute, by_attribute_mixed, adios2)");
    }

    // Copy attenuation params (may have been overwritten in Load* functions)
    data.attenuation = ConvertAttenuation(config.attenuation);

    return data;
}

// =============================================================================
// Private: Format-specific loaders
// =============================================================================

// Helper to get value from params map with default
static real_t GetParam(const std::map<std::string, real_t>& params,
                       const std::string& key, real_t default_val = 0.0) {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
}

IsotropicElasticInput IsotropicElasticLoader::LoadConstant(const MaterialConfig& config) {
    IsotropicElasticInput data;
    data.source = DataSource::Constant;

    // Extract isotropic elastic parameters from generic params map
    IsotropicElasticConstantParams params;
    params.vp = GetParam(config.params, "vp");
    params.vs = GetParam(config.params, "vs");
    params.rho = GetParam(config.params, "rho");

    // Validate required parameters
    MFEM_VERIFY(params.vp > 0 && params.vs > 0 && params.rho > 0,
        "Isotropic elastic material requires positive vp, vs, and rho");

    // Q values from attenuation config (if enabled)
    if (config.attenuation.enabled) {
        params.qkappa = config.attenuation.qkappa;
        params.qmu = config.attenuation.qmu;
    }

    data.constant = params;
    return data;
}

IsotropicElasticInput IsotropicElasticLoader::LoadGrid2D(const MaterialConfig& config) {
    IsotropicElasticInput data;
    data.source = DataSource::Grid;
    data.is_3d = false;

    IsotropicElasticMaterialData2D grid_data;
    bool read_q = config.attenuation.enabled;

    bool success = ReadIsotropicElasticMaterialData2D(
        config.material_file, grid_data, read_q);

    if (!success) {
        MFEM_ABORT("Failed to read 2D elastic material file: " + config.material_file);
    }

    data.grid_2d = std::move(grid_data);
    return data;
}

IsotropicElasticInput IsotropicElasticLoader::LoadGrid3D(const MaterialConfig& config) {
    IsotropicElasticInput data;
    data.source = DataSource::Grid;
    data.is_3d = true;

    IsotropicElasticMaterialData3D grid_data;
    bool read_q = config.attenuation.enabled;

    bool success = ReadIsotropicElasticMaterialData3D(
        config.material_file, grid_data, read_q);

    if (!success) {
        MFEM_ABORT("Failed to read 3D elastic material file: " + config.material_file);
    }

    data.grid_3d = std::move(grid_data);
    return data;
}

IsotropicElasticInput IsotropicElasticLoader::LoadByAttribute(const MaterialConfig& config) {
    IsotropicElasticInput data;
    data.source = DataSource::ByAttribute;

    std::vector<IsotropicElasticAttributeEntry> entries;
    bool read_q = config.attenuation.enabled;

    bool success = ReadIsotropicElasticAttributeMaterial(
        config.by_attribute_file, entries, read_q);

    if (!success) {
        MFEM_ABORT("Failed to read elastic attribute material file: " + config.by_attribute_file);
    }

    if (entries.empty()) {
        MFEM_ABORT("No entries found in attribute material file: " + config.by_attribute_file);
    }

    data.by_attribute = std::move(entries);
    return data;
}

IsotropicElasticInput IsotropicElasticLoader::LoadByAttributeMixed(
    const MaterialConfig& config, bool is_3d)
{
    IsotropicElasticInput data;
    data.source = DataSource::ByAttributeMixed;
    data.is_3d = is_3d;

    std::vector<IsotropicElasticMixedAttributeEntry> entries;

    for (const auto& cfg_entry : config.attribute_entries) {
        IsotropicElasticMixedAttributeEntry entry;
        entry.attribute = cfg_entry.attribute;

        entry.mode = cfg_entry.mode;

        if (cfg_entry.mode == "constant") {
            entry.is_heterogeneous = false;

            // Extract isotropic elastic params from generic params map
            entry.vp = GetParam(cfg_entry.params, "vp");
            entry.vs = GetParam(cfg_entry.params, "vs");
            entry.rho = GetParam(cfg_entry.params, "rho");

            // Validate required parameters
            MFEM_VERIFY(entry.vp > 0 && entry.vs > 0 && entry.rho > 0,
                "Attribute " << entry.attribute
                << ": isotropic elastic requires positive vp, vs, and rho");

            // Q values (optional)
            entry.Qkappa = GetParam(cfg_entry.params, "qkappa", -1.0);
            entry.Qmu = GetParam(cfg_entry.params, "qmu", -1.0);
        }
        else if (cfg_entry.mode == "grid") {
            entry.is_heterogeneous = true;

            // Load grid data from file
            bool read_q = config.attenuation.enabled;

            if (is_3d) {
                IsotropicElasticMaterialData3D grid_data;
                bool success = ReadIsotropicElasticMaterialData3D(
                    cfg_entry.grid_file, grid_data, read_q);
                MFEM_VERIFY(success,
                    "Failed to read 3D elastic material file for attribute "
                    << entry.attribute << ": " << cfg_entry.grid_file);
                entry.grid_data_3d = std::move(grid_data);
            } else {
                IsotropicElasticMaterialData2D grid_data;
                bool success = ReadIsotropicElasticMaterialData2D(
                    cfg_entry.grid_file, grid_data, read_q);
                MFEM_VERIFY(success,
                    "Failed to read 2D elastic material file for attribute "
                    << entry.attribute << ": " << cfg_entry.grid_file);
                entry.grid_data_2d = std::move(grid_data);
            }
        }
        else if (cfg_entry.mode == "adios2") {
            entry.is_heterogeneous = true;  // adios2 is heterogeneous in nature

            MPI_Comm comm = MPI_COMM_WORLD;
            MFEM_VERIFY(!cfg_entry.adios2_vp_file.empty(),
                "Attribute " << entry.attribute
                << " (adios2) requires vp_file");
            MFEM_VERIFY(!cfg_entry.adios2_vs_file.empty(),
                "Attribute " << entry.attribute
                << " (adios2) requires vs_file (elastic)");
            MFEM_VERIFY(!cfg_entry.adios2_rho_file.empty(),
                "Attribute " << entry.attribute
                << " (adios2) requires rho_file");

            entry.adios2_vp = std::make_shared<MaterialField>(
                LoadFieldBP(cfg_entry.adios2_vp_file, "data", comm));
            entry.adios2_vs = std::make_shared<MaterialField>(
                LoadFieldBP(cfg_entry.adios2_vs_file, "data", comm));
            entry.adios2_rho = std::make_shared<MaterialField>(
                LoadFieldBP(cfg_entry.adios2_rho_file, "data", comm));

            if (config.attenuation.enabled) {
                if (!cfg_entry.adios2_qkappa_file.empty()) {
                    entry.adios2_qkappa = std::make_shared<MaterialField>(
                        LoadFieldBP(cfg_entry.adios2_qkappa_file, "data", comm));
                }
                if (!cfg_entry.adios2_qmu_file.empty()) {
                    entry.adios2_qmu = std::make_shared<MaterialField>(
                        LoadFieldBP(cfg_entry.adios2_qmu_file, "data", comm));
                }
            }
        }
        else {
            MFEM_ABORT("Attribute " << entry.attribute
                       << ": unknown mode '" << cfg_entry.mode
                       << "' (supported: constant, grid, adios2)");
        }

        entries.push_back(std::move(entry));
    }

    MFEM_VERIFY(!entries.empty(),
        "No attribute entries provided for by_attribute_mixed format");

    data.by_attribute_mixed = std::move(entries);
    return data;
}

IsotropicElasticInput IsotropicElasticLoader::LoadADIOS2(
    const MaterialConfig& config, bool is_3d)
{
    IsotropicElasticInput data;
    data.source = DataSource::ADIOS2;
    data.is_3d = is_3d;

    MPI_Comm comm = MPI_COMM_WORLD;

    MFEM_VERIFY(!config.adios2_vp_file.empty(),
        "ADIOS2 elastic material requires vp_file");
    MFEM_VERIFY(!config.adios2_vs_file.empty(),
        "ADIOS2 elastic material requires vs_file");
    MFEM_VERIFY(!config.adios2_rho_file.empty(),
        "ADIOS2 elastic material requires rho_file");

    data.adios2_vp = std::make_shared<MaterialField>(
        LoadFieldBP(config.adios2_vp_file, "data", comm));
    data.adios2_vs = std::make_shared<MaterialField>(
        LoadFieldBP(config.adios2_vs_file, "data", comm));
    data.adios2_rho = std::make_shared<MaterialField>(
        LoadFieldBP(config.adios2_rho_file, "data", comm));

    // Load Q fields if attenuation is enabled and files are specified
    if (config.attenuation.enabled) {
        if (!config.adios2_qkappa_file.empty()) {
            data.adios2_qkappa = std::make_shared<MaterialField>(
                LoadFieldBP(config.adios2_qkappa_file, "data", comm));
        }
        if (!config.adios2_qmu_file.empty()) {
            data.adios2_qmu = std::make_shared<MaterialField>(
                LoadFieldBP(config.adios2_qmu_file, "data", comm));
        }
    }

    return data;
}

// =============================================================================
// Private: Helper functions
// =============================================================================

MaterialAttenuationConfig IsotropicElasticLoader::ConvertAttenuation(const AttenuationConfig& config) {
    MaterialAttenuationConfig params;
    params.enabled = config.enabled;
    params.f0 = config.f0;
    params.n_units = config.n_units;
    params.qkappa = config.qkappa;
    params.qmu = config.qmu;
    return params;
}

}  // namespace SEM
