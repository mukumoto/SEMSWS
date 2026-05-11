/**
 * @file IsotropicAcousticLoader.cpp
 * @brief Implementation of IsotropicAcousticLoader
 */

#include "material/isotropic_acoustic/IsotropicAcousticLoader.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticIO.hpp"
#include "io/ADIOS2IO.hpp"
#include <mfem.hpp>

namespace SEM {

// Helper to get value from params map with default
static real_t GetParam(const std::map<std::string, real_t>& params,
                       const std::string& key, real_t default_val = 0.0) {
    auto it = params.find(key);
    return (it != params.end()) ? it->second : default_val;
}

IsotropicAcousticInput IsotropicAcousticLoader::Load(const MaterialConfig& config, bool is_3d) {
    IsotropicAcousticInput data;
    data.is_3d = is_3d;
    data.attenuation = ConvertAttenuation(config.attenuation);

    if (config.format == "constant") {
        data = LoadConstant(config);
        data.is_3d = is_3d;
    }
    else if (config.format == "grid") {
        data = is_3d ? LoadGrid3D(config) : LoadGrid2D(config);
    }
    else if (config.format == "by_attribute") {
        data = LoadByAttribute(config);
        data.is_3d = is_3d;
    }
    else if (config.format == "by_attribute_mixed") {
        data = LoadByAttributeMixed(config, is_3d);
    }
    else if (config.format == "adios2") {
        data = LoadADIOS2(config, is_3d);
    }
    else {
        MFEM_ABORT("Unsupported acoustic material format: " + config.format +
                   " (supported: constant, grid, by_attribute, by_attribute_mixed, adios2)");
    }

    data.attenuation = ConvertAttenuation(config.attenuation);
    return data;
}

IsotropicAcousticInput IsotropicAcousticLoader::LoadConstant(const MaterialConfig& config) {
    IsotropicAcousticInput data;
    data.source = DataSource::Constant;

    // Extract acoustic parameters from generic params map
    IsotropicAcousticConstantParams params;
    params.vp = GetParam(config.params, "vp");
    params.rho = GetParam(config.params, "rho");

    // Validate required parameters
    MFEM_VERIFY(params.vp > 0 && params.rho > 0,
        "Isotropic acoustic material requires positive vp and rho");

    // Q value from attenuation config (if enabled)
    if (config.attenuation.enabled) {
        params.qkappa = config.attenuation.qkappa;
    }

    data.constant = params;
    return data;
}

IsotropicAcousticInput IsotropicAcousticLoader::LoadGrid2D(const MaterialConfig& config) {
    IsotropicAcousticInput data;
    data.source = DataSource::Grid;
    data.is_3d = false;

    IsotropicAcousticMaterialData2D grid_data;
    bool read_q = config.attenuation.enabled;

    bool success = ReadIsotropicAcousticMaterialData2D(
        config.material_file, grid_data, read_q);

    if (!success) {
        MFEM_ABORT("Failed to read 2D acoustic material file: " + config.material_file);
    }

    data.grid_2d = std::move(grid_data);
    return data;
}

IsotropicAcousticInput IsotropicAcousticLoader::LoadGrid3D(const MaterialConfig& config) {
    IsotropicAcousticInput data;
    data.source = DataSource::Grid;
    data.is_3d = true;

    IsotropicAcousticMaterialData3D grid_data;
    bool read_q = config.attenuation.enabled;

    bool success = ReadIsotropicAcousticMaterialData3D(
        config.material_file, grid_data, read_q);

    if (!success) {
        MFEM_ABORT("Failed to read 3D acoustic material file: " + config.material_file);
    }

    data.grid_3d = std::move(grid_data);
    return data;
}

IsotropicAcousticInput IsotropicAcousticLoader::LoadByAttribute(const MaterialConfig& config) {
    IsotropicAcousticInput data;
    data.source = DataSource::ByAttribute;

    std::vector<IsotropicAcousticAttributeEntry> entries;
    bool read_q = config.attenuation.enabled;

    bool success = ReadIsotropicAcousticAttributeMaterial(
        config.by_attribute_file, entries, read_q);

    if (!success) {
        MFEM_ABORT("Failed to read acoustic attribute material file: " + config.by_attribute_file);
    }

    if (entries.empty()) {
        MFEM_ABORT("No entries found in acoustic attribute material file: " + config.by_attribute_file);
    }

    data.by_attribute = std::move(entries);
    return data;
}

IsotropicAcousticInput IsotropicAcousticLoader::LoadByAttributeMixed(
    const MaterialConfig& config, bool is_3d)
{
    IsotropicAcousticInput data;
    data.source = DataSource::ByAttributeMixed;
    data.is_3d = is_3d;

    std::vector<IsotropicAcousticMixedAttributeEntry> entries;

    for (const auto& cfg_entry : config.attribute_entries) {
        IsotropicAcousticMixedAttributeEntry entry;
        entry.attribute = cfg_entry.attribute;

        entry.mode = cfg_entry.mode;

        if (cfg_entry.mode == "constant") {
            entry.is_heterogeneous = false;

            // Extract acoustic params from generic params map
            entry.vp = GetParam(cfg_entry.params, "vp");
            entry.rho = GetParam(cfg_entry.params, "rho");

            // Validate required parameters
            MFEM_VERIFY(entry.vp > 0 && entry.rho > 0,
                "Attribute " << entry.attribute
                << ": isotropic acoustic requires positive vp and rho");

            // Q value (optional)
            entry.Qkappa = GetParam(cfg_entry.params, "qkappa", -1.0);
        }
        else if (cfg_entry.mode == "grid") {
            entry.is_heterogeneous = true;

            // Load grid data from file
            bool read_q = config.attenuation.enabled;

            if (is_3d) {
                IsotropicAcousticMaterialData3D grid_data;
                bool success = ReadIsotropicAcousticMaterialData3D(
                    cfg_entry.grid_file, grid_data, read_q);
                MFEM_VERIFY(success,
                    "Failed to read 3D acoustic material file for attribute "
                    << entry.attribute << ": " << cfg_entry.grid_file);
                entry.grid_data_3d = std::move(grid_data);
            } else {
                IsotropicAcousticMaterialData2D grid_data;
                bool success = ReadIsotropicAcousticMaterialData2D(
                    cfg_entry.grid_file, grid_data, read_q);
                MFEM_VERIFY(success,
                    "Failed to read 2D acoustic material file for attribute "
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
            MFEM_VERIFY(!cfg_entry.adios2_rho_file.empty(),
                "Attribute " << entry.attribute
                << " (adios2) requires rho_file");

            entry.adios2_vp = std::make_shared<MaterialField>(
                LoadFieldBP(cfg_entry.adios2_vp_file, "data", comm));
            entry.adios2_rho = std::make_shared<MaterialField>(
                LoadFieldBP(cfg_entry.adios2_rho_file, "data", comm));

            if (config.attenuation.enabled
                && !cfg_entry.adios2_qkappa_file.empty()) {
                entry.adios2_qkappa = std::make_shared<MaterialField>(
                    LoadFieldBP(cfg_entry.adios2_qkappa_file, "data", comm));
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

IsotropicAcousticInput IsotropicAcousticLoader::LoadADIOS2(
    const MaterialConfig& config, bool is_3d)
{
    IsotropicAcousticInput data;
    data.source = DataSource::ADIOS2;
    data.is_3d = is_3d;

    MPI_Comm comm = MPI_COMM_WORLD;

    MFEM_VERIFY(!config.adios2_vp_file.empty(),
        "ADIOS2 acoustic material requires vp_file");
    MFEM_VERIFY(!config.adios2_rho_file.empty(),
        "ADIOS2 acoustic material requires rho_file");

    data.adios2_vp = std::make_shared<MaterialField>(
        LoadFieldBP(config.adios2_vp_file, "data", comm));
    data.adios2_rho = std::make_shared<MaterialField>(
        LoadFieldBP(config.adios2_rho_file, "data", comm));

    // Load Qkappa if attenuation is enabled and file is specified
    if (config.attenuation.enabled && !config.adios2_qkappa_file.empty()) {
        data.adios2_qkappa = std::make_shared<MaterialField>(
            LoadFieldBP(config.adios2_qkappa_file, "data", comm));
    }

    return data;
}

MaterialAttenuationConfig IsotropicAcousticLoader::ConvertAttenuation(const AttenuationConfig& config) {
    MaterialAttenuationConfig params;
    params.enabled = config.enabled;
    params.f0 = config.f0;
    params.n_units = config.n_units;
    params.qkappa = config.qkappa;
    params.qmu = 0;  // Acoustic has no Qmu
    return params;
}

}  // namespace SEM
