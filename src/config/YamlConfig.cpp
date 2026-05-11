/**
 * @file YamlConfig.cpp
 * @brief Implementation of YAML configuration parser for SEMSWS
 */

#include "config/YamlConfig.hpp"
#include "common/Types.hpp"
#include "common/BoundaryUtils.hpp"
#include "srcrecv/HDF5SourceReceiverReader.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <vector>

namespace SEM {

namespace {

/// Reject any key in `node` that is not present in `allowed`. Used as a
/// strict-key guard by block-level parsers/validators so typos, renamed
/// keys, and removed-legacy forms become hard errors instead of silent
/// no-ops. Returns false and writes to `error_msg` on the first unknown
/// key; leaves `error_msg` untouched on success. When `node` is not a
/// mapping (scalar / null / sequence) there are no keys to check and the
/// function trivially succeeds.
bool CheckKnownKeys(const YAML::Node& node,
                    const std::set<std::string>& allowed,
                    const std::string& context,
                    std::string& error_msg) {
    if (!node || !node.IsMap()) return true;
    for (const auto& kv : node) {
        const std::string key = kv.first.as<std::string>();
        if (!allowed.count(key)) {
            std::ostringstream oss;
            oss << "Unknown key `" << key << "` in " << context << " (allowed: ";
            bool first = true;
            for (const auto& k : allowed) {
                if (!first) oss << ", ";
                oss << k;
                first = false;
            }
            oss << ")";
            error_msg = oss.str();
            return false;
        }
    }
    return true;
}

/// MFEM_ABORT-on-failure variant for parsers that are already in the
/// abort-on-error path (ApplyWavefieldNode, ParseOutputFormat, etc.).
void RequireKnownKeys(const YAML::Node& node,
                      const std::set<std::string>& allowed,
                      const std::string& context) {
    std::string err;
    if (!CheckKnownKeys(node, allowed, context, err)) {
        MFEM_ABORT(err);
    }
}

/// Parse a single material sub-node (as used under `material.fluid` and
/// `material.solid` for `type: coupled`). Mirrors the flat LoadMaterialConfig
/// path in ConfigLoaders.cpp but drives off a YAML::Node rather than the
/// flat YamlConfig getters, so the same logic can be reused per-submesh.
MaterialConfig ParseMaterialSubNode(const YAML::Node& node,
                                    const std::string& sub_label)
{
    MFEM_VERIFY(node, "material." << sub_label << " block is missing");

    MaterialConfig mc;

    MFEM_VERIFY(node["type"],   "material." << sub_label << ".type is required");
    MFEM_VERIFY(node["format"], "material." << sub_label << ".format is required");
    mc.material_type = node["type"].as<std::string>();
    mc.format        = node["format"].as<std::string>();

    const auto mat_type = StringToMaterialType(mc.material_type);  // aborts if unknown
    MFEM_VERIFY(mat_type != MaterialType::Coupled,
                "nested material." << sub_label << ".type cannot itself be 'coupled'");

    if (mc.format == "constant") {
        if (mat_type == MaterialType::IsotropicAcoustic) {
            MFEM_VERIFY(node["vp"] && node["rho"],
                        "material." << sub_label << " (constant acoustic) requires vp, rho");
            mc.params["vp"]  = node["vp"].as<real_t>();
            mc.params["rho"] = node["rho"].as<real_t>();
        } else if (mat_type == MaterialType::IsotropicElastic) {
            MFEM_VERIFY(node["vp"] && node["vs"] && node["rho"],
                        "material." << sub_label << " (constant elastic) requires vp, vs, rho");
            mc.params["vp"]  = node["vp"].as<real_t>();
            mc.params["vs"]  = node["vs"].as<real_t>();
            mc.params["rho"] = node["rho"].as<real_t>();
        } else {
            MFEM_ABORT("material." << sub_label << ".format=constant is not supported "
                       "for material_type=" << mc.material_type);
        }
    } else if (mc.format == "grid" || mc.format == "by_attribute" ||
               mc.format == "by_attribute_mixed") {
        MFEM_VERIFY(node["file"],
                    "material." << sub_label << ".format=" << mc.format
                    << " requires 'file'");
        if (mc.format == "grid") {
            mc.material_file = node["file"].as<std::string>();
        } else {
            mc.by_attribute_file = node["file"].as<std::string>();
        }
    } else {
        MFEM_ABORT("material." << sub_label << ".format=" << mc.format
                   << " is not yet supported for coupled materials");
    }

    // Parse nested attenuation block, if any. Same shape as the top-level
    // `material.attenuation` block handled by `Validate()` below — f0 +
    // n_units mandatory, Qkappa/Qmu required for the `constant` format
    // (read from the Q files for grid / by_attribute modes).
    if (node["attenuation"] &&
        node["attenuation"]["enabled"] &&
        node["attenuation"]["enabled"].as<bool>())
    {
        const YAML::Node atten = node["attenuation"];
        MFEM_VERIFY(atten["f0"],
                    "material." << sub_label << ".attenuation.f0 is required");
        MFEM_VERIFY(atten["n_units"],
                    "material." << sub_label << ".attenuation.n_units is required");

        mc.attenuation.enabled = true;
        mc.attenuation.f0      = atten["f0"].as<real_t>();
        mc.attenuation.n_units = atten["n_units"].as<int>();

        if (mc.format == "constant") {
            MFEM_VERIFY(atten["Qkappa"],
                        "material." << sub_label
                        << ".attenuation.Qkappa is required for format=constant");
            mc.attenuation.qkappa = atten["Qkappa"].as<real_t>();
            if (mat_type == MaterialType::IsotropicElastic) {
                MFEM_VERIFY(atten["Qmu"],
                            "material." << sub_label
                            << ".attenuation.Qmu is required for format=constant elastic");
                mc.attenuation.qmu = atten["Qmu"].as<real_t>();
            }
        }
        // For grid / by_attribute{,_mixed}: Q fields are read from the
        // material file alongside vp/vs/rho, so no extra parsing here.
    }

    return mc;
}

/// Validate that a nested material sub-block (under `material.fluid` or
/// `material.solid`) has the fields that ParseMaterialSubNode will read.
/// Writes the error message on failure and returns false.
bool ValidateMaterialSubNode(const YAML::Node& node,
                             const std::string& sub_label,
                             std::string& error_msg)
{
    if (!node) {
        error_msg = "Missing required section: material." + sub_label
                  + " (required when material.type = coupled)";
        return false;
    }
    if (!CheckKnownKeys(node, {
            "attribute", "type", "format",
            "vp", "vs", "rho", "file",
            "attenuation"
        }, "material." + sub_label, error_msg)) {
        return false;
    }
    if (node["attenuation"] &&
        !CheckKnownKeys(node["attenuation"], {
            "enabled", "f0", "n_units",
            "Qkappa", "Qmu", "Qkappa_file", "Qmu_file"
        }, "material." + sub_label + ".attenuation", error_msg)) {
        return false;
    }
    if (!node["attribute"]) {
        error_msg = "Missing required parameter: material." + sub_label + ".attribute";
        return false;
    }
    if (!node["type"]) {
        error_msg = "Missing required parameter: material." + sub_label + ".type";
        return false;
    }
    if (!node["format"]) {
        error_msg = "Missing required parameter: material." + sub_label + ".format";
        return false;
    }
    const auto mat_type = StringToMaterialType(node["type"].as<std::string>());
    if (mat_type == MaterialType::Coupled) {
        error_msg = "material." + sub_label
                  + ".type must be a concrete material, not 'coupled'";
        return false;
    }
    const std::string fmt = node["format"].as<std::string>();
    if (fmt == "constant") {
        if (mat_type == MaterialType::IsotropicAcoustic) {
            if (!node["vp"] || !node["rho"]) {
                error_msg = "material." + sub_label
                          + " (constant acoustic) requires vp and rho";
                return false;
            }
        } else if (mat_type == MaterialType::IsotropicElastic) {
            if (!node["vp"] || !node["vs"] || !node["rho"]) {
                error_msg = "material." + sub_label
                          + " (constant elastic) requires vp, vs and rho";
                return false;
            }
        } else {
            error_msg = "material." + sub_label
                      + ".format=constant not supported for type="
                      + node["type"].as<std::string>();
            return false;
        }
    }
    // grid / by_attribute / by_attribute_mixed: file is required
    else if (fmt == "grid" || fmt == "by_attribute" || fmt == "by_attribute_mixed") {
        if (!node["file"]) {
            error_msg = "material." + sub_label + ".format=" + fmt
                      + " requires 'file'";
            return false;
        }
    } else {
        error_msg = "material." + sub_label + ".format=" + fmt
                  + " is not yet supported for coupled materials";
        return false;
    }

    // Attenuation block (optional). When enabled, f0 / n_units are
    // always required; Qkappa (+ Qmu for elastic) required for format=constant.
    if (node["attenuation"] &&
        node["attenuation"]["enabled"] &&
        node["attenuation"]["enabled"].as<bool>())
    {
        const YAML::Node atten = node["attenuation"];
        if (!atten["f0"]) {
            error_msg = "material." + sub_label
                      + ".attenuation.f0 is required when enabled";
            return false;
        }
        if (!atten["n_units"]) {
            error_msg = "material." + sub_label
                      + ".attenuation.n_units is required when enabled";
            return false;
        }
        if (fmt == "constant") {
            if (!atten["Qkappa"]) {
                error_msg = "material." + sub_label
                          + ".attenuation.Qkappa is required for format=constant";
                return false;
            }
            if (mat_type == MaterialType::IsotropicElastic && !atten["Qmu"]) {
                error_msg = "material." + sub_label
                          + ".attenuation.Qmu is required for format=constant elastic";
                return false;
            }
        }
    }
    return true;
}

}  // namespace

// =============================================================================
// Constructor
// =============================================================================

YamlConfig::YamlConfig()
    : valid_(false), sources_parsed_(false), receivers_parsed_(false) {
    error_msg_ = "No configuration file loaded";
}

YamlConfig::YamlConfig(const std::string& filepath)
    : filepath_(filepath), valid_(false), sources_parsed_(false), receivers_parsed_(false) {
    try {
        root_ = YAML::LoadFile(filepath);
        Validate();
    } catch (const YAML::Exception& e) {
        valid_ = false;
        // Enhance error message: if yaml-cpp reports a mark (line/col),
        // re-open the file and dump a ±2-line context window with a caret
        // pointing at the offending column. Dramatically shortens the
        // debug loop for typos / wrong indentation.
        std::ostringstream oss;
        oss << "YAML parsing error: " << e.msg;
        if (e.mark.line >= 0) {
            oss << " (line " << (e.mark.line + 1)
                << ", col " << (e.mark.column + 1) << ")";
            std::ifstream ifs(filepath);
            if (ifs) {
                std::vector<std::string> lines;
                std::string ln;
                while (std::getline(ifs, ln)) lines.push_back(ln);
                const int err = e.mark.line;
                const int beg = std::max(0, err - 2);
                const int end = std::min(static_cast<int>(lines.size()),
                                         err + 3);
                oss << "\n";
                for (int i = beg; i < end; ++i) {
                    oss << (i == err ? "--> " : "    ")
                        << std::setw(4) << (i + 1) << " | "
                        << lines[i] << "\n";
                    if (i == err) {
                        oss << "         " << std::string(e.mark.column, ' ')
                            << "^\n";
                    }
                }
            }
        }
        error_msg_ = oss.str();
    } catch (const std::exception& e) {
        valid_ = false;
        error_msg_ = "Error loading config: " + std::string(e.what());
    }
}

// =============================================================================
// Validation
// =============================================================================

void YamlConfig::Validate() {
    valid_ = true;
    error_msg_.clear();

    // Strict-key validation on the top-level document. Any key not in
    // this set is a typo or removed-legacy form and must abort rather
    // than silently pass through. Add new top-level keys here when
    // introducing them; do NOT accept unknown siblings.
    if (!CheckKnownKeys(root_, {
            "name", "description",
            "simulation", "mesh", "material", "boundary",
            "sources", "receivers", "device", "inversion"
        }, "root", error_msg_)) {
        valid_ = false;
        return;
    }

    //name and description
    if (!root_["name"]) {
        root_["name"] = "Seismic wave simulation";
    }

    if(!root_["description"]) {
        root_["description"] = "No description provided.";
    }

    //=============================================================================
    // Validate Simulation Section
    //=============================================================================
    YAML::Node sim = root_["simulation"];
    if (!root_["simulation"]) {
        error_msg_ = "Missing required section: simulation";
        valid_ = false;
        return;
    }

    // Strict-key validation for `simulation` — rejects the removed
    // `rank0_distribution` and any other stale/mistyped key.
    if (!CheckKnownKeys(sim, {
            "dimension", "order", "time", "output",
            "mode", "misfit_type", "num_checkpoints",
            "checkpoint_storage", "checkpoint_device", "checkpoint_dir",
            "kernel_output_dir"
        }, "simulation", error_msg_)) {
        valid_ = false;
        return;
    }

    // simulation.dimension
    if (!sim["dimension"]) {
        error_msg_ = "Missing required parameter: simulation.dimension";
        valid_ = false;
        return;
    }

    int dim = sim["dimension"].as<int>();
    if (dim != 2 && dim != 3) {
        error_msg_ = "Invalid dimension: must be 2 or 3";
        valid_ = false;
        return;
    }

    // simulation.order
    if (!sim["order"]) {
        error_msg_ = "Missing required parameter: simulation.order";
        valid_ = false;
        return;
    }
    int order = sim["order"].as<int>();
    if (order < 1) {
        error_msg_ = "Invalid simulation.order: must be >= 1";
        valid_ = false;
        return;
    }

    // simulation.time section
    if (!sim["time"]) {
        error_msg_ = "Missing required section: simulation.time";
        valid_ = false;
        return;
    }

    YAML::Node time_node = sim["time"];
    if (!CheckKnownKeys(time_node,
            {"dt", "steps", "cfl_factor", "t0"},
            "simulation.time", error_msg_)) {
        valid_ = false;
        return;
    }
    // simulation.time.dt
    if (!time_node["dt"]) {
        error_msg_ = "Missing required parameter: simulation.time.dt";
        valid_ = false;
        return;
    }
    auto dt = time_node["dt"].as<real_t>();
    if (dt <= 0.0) {
        error_msg_ = "Invalid simulation.time.dt: must be positive";
        valid_ = false;
        return;
    }
    // simulation.time.steps
    if (!time_node["steps"]) {
        error_msg_ = "Missing required parameter: simulation.time.steps";
        valid_ = false;
        return;
    }
    auto steps = time_node["steps"].as<int>();
    if (steps <= 0) {
        error_msg_ = "Invalid simulation.time.steps: must be positive";
        valid_ = false;
        return;
    }
    // cfl factor
    if (!time_node["cfl_factor"]) {
        error_msg_ = "Missing required parameter: simulation.time.cfl_factor";
        valid_ = false;
        return;
    }
    auto cfl = time_node["cfl_factor"].as<real_t>();
    if (cfl <= 0.0 || cfl > 0.7) {
        error_msg_ = "Invalid simulation.time.cfl_factor: must be in (0.0, 0.7]";
        valid_ = false;
        return;
    }

    // simulation.output section
    if (!sim["output"]) {
        error_msg_ = "Missing required section: simulation.output";
        valid_ = false;
        return;
    }
    YAML::Node output_node = sim["output"];
    if (!CheckKnownKeys(output_node,
            {"directory", "log_interval", "wavefield", "material",
             "summary_file"},
            "simulation.output", error_msg_)) {
        valid_ = false;
        return;
    }
    // simulation.output.directory
    if (!output_node["directory"]) {
        error_msg_ = "Missing required parameter: simulation.output.directory";
        valid_ = false;
        return;
    }
    // simulation.output.log_interval
    if (!output_node["log_interval"]) {
        error_msg_ = "Missing required parameter: simulation.output.log_interval";
        valid_ = false;
        return;
    }
    // Strict-key check on the wavefield block (if present). Per-side
    // `fluid:` / `solid:` sub-blocks are validated recursively below.
    if (output_node["wavefield"]) {
        if (!CheckKnownKeys(output_node["wavefield"],
                {"enabled", "interval", "fields", "formats", "components",
                 "fluid", "solid"},
                "simulation.output.wavefield", error_msg_)) {
            valid_ = false;
            return;
        }
        for (const char* side : {"fluid", "solid"}) {
            if (output_node["wavefield"][side] &&
                !CheckKnownKeys(output_node["wavefield"][side],
                    {"enabled", "interval", "fields", "formats", "components"},
                    std::string("simulation.output.wavefield.") + side,
                    error_msg_)) {
                valid_ = false;
                return;
            }
        }
    }
    if (output_node["material"]) {
        if (!CheckKnownKeys(output_node["material"],
                {"enabled", "fields", "formats", "fluid", "solid"},
                "simulation.output.material", error_msg_)) {
            valid_ = false;
            return;
        }
        for (const char* side : {"fluid", "solid"}) {
            if (output_node["material"][side] &&
                !CheckKnownKeys(output_node["material"][side],
                    {"enabled", "fields", "formats"},
                    std::string("simulation.output.material.") + side,
                    error_msg_)) {
                valid_ = false;
                return;
            }
        }
    }
    // Validate wavefield output parameters when enabled
    if (output_node["wavefield"] && output_node["wavefield"]["enabled"] &&
        output_node["wavefield"]["enabled"].as<bool>()) {
        YAML::Node wf = output_node["wavefield"];
        if (!wf["interval"]) {
            error_msg_ = "Missing required parameter: simulation.output.wavefield.interval (when wavefield enabled)";
            valid_ = false;
            return;
        }
        if (wf["format"]) {
            error_msg_ = "simulation.output.wavefield.format has been "
                         "removed. Use `formats: [{type: <name>}, ...]` "
                         "(mapping list).";
            valid_ = false;
            return;
        }
        if (!wf["formats"]) {
            error_msg_ = "Missing required parameter: "
                         "simulation.output.wavefield.formats "
                         "(when wavefield enabled)";
            valid_ = false;
            return;
        }
    }

    //=============================================================================
    // Validate Mesh Section
    //=============================================================================
    if (!root_["mesh"]) {
        error_msg_ = "Missing required section: mesh";
        valid_ = false;
        return;
    }
    YAML::Node mesh = root_["mesh"];
    if (!CheckKnownKeys(mesh, {
            "type", "file", "format",
            "origin", "size", "elements",
            "max_freq", "ppw",
            "save", "partition", "partition_grid", "partitioned",
            "attr_y_threshold"
        }, "mesh", error_msg_)) {
        valid_ = false;
        return;
    }

    // mesh.type
    if (!mesh["type"]) {
        error_msg_ = "Missing required parameter: mesh.type";
        valid_ = false;
        return;
    }

    std::string mesh_type = mesh["type"].as<std::string>();

    if (mesh_type == "external") {
        if (!mesh["file"]) {
            error_msg_ = "External mesh requires 'file' parameter";
            valid_ = false;
            return;
        }
        // mesh.format is optional - MFEM auto-detects from file header
    } else if (mesh_type == "internal") {
        if (!mesh["origin"] || !mesh["size"] || !mesh["elements"]) {
            error_msg_ = "Internal mesh requires 'origin', 'size', and 'elements' parameters";
            valid_ = false;
            return;
        }
        else{
            //check the dimensions of origin, size, elements
            YAML::Node origin = mesh["origin"];
            YAML::Node size = mesh["size"];
            YAML::Node elements = mesh["elements"];
            if(origin.size() != dim || size.size() != dim || elements.size() != dim){
                error_msg_ = "Internal mesh 'origin', 'size', and 'elements' must have length equal to dimension";
                valid_ = false;
                return;
            }
        }
    } else if (mesh_type == "partitioned") {
        // Pre-partitioned mesh files for memory-efficient loading
        YAML::Node partitioned = mesh["partitioned"];
        if (!partitioned) {
            error_msg_ = "Partitioned mesh requires 'partitioned' section";
            valid_ = false;
            return;
        }
        if (!CheckKnownKeys(partitioned,
                {"directory", "nparts"},
                "mesh.partitioned", error_msg_)) {
            valid_ = false;
            return;
        }
        if (!partitioned["directory"]) {
            error_msg_ = "Partitioned mesh requires 'partitioned.directory' parameter";
            valid_ = false;
            return;
        }
        if (!partitioned["nparts"]) {
            error_msg_ = "Partitioned mesh requires 'partitioned.nparts' parameter";
            valid_ = false;
            return;
        }
        int nparts = partitioned["nparts"].as<int>();
        if (nparts < 1) {
            error_msg_ = "partitioned.nparts must be >= 1";
            valid_ = false;
            return;
        }
    } else {
        error_msg_ = "Invalid mesh type: " + mesh_type + " (must be external, internal, or partitioned)";
        valid_ = false;
        return;
    }


    // mesh.max_freq (required)
    if (!mesh["max_freq"]) {
        error_msg_ = "Missing required parameter: mesh.max_freq";
        valid_ = false;
        return;
    }
    if (mesh["max_freq"].as<real_t>() <= 0.0) {
        error_msg_ = "mesh.max_freq must be > 0";
        valid_ = false;
        return;
    }

    // mesh.ppw (required)
    if (!mesh["ppw"]) {
        error_msg_ = "Missing required parameter: mesh.ppw";
        valid_ = false;
        return;
    }
    if (mesh["ppw"].as<real_t>() <= 0.0) {
        error_msg_ = "mesh.ppw must be > 0";
        valid_ = false;
        return;
    }

    //=============================================================================
    // Validate Material Section
    //=============================================================================
    YAML::Node mat = root_["material"];
    if (!mat) {
        error_msg_ = "Missing required section: material";
        valid_ = false;
        return;
    }
    // Material strict-key validation accepts the union of flat and
    // coupled keys; the specific flat/coupled path below enforces which
    // of those keys are required for the chosen type.
    if (!CheckKnownKeys(mat, {
            "type", "format", "file",
            "vp", "vs", "rho",
            "vp_file", "vs_file", "rho_file", "qkappa_file", "qmu_file",
            "attenuation",
            "export_model", "export_dir",
            "fluid", "solid"
        }, "material", error_msg_)) {
        valid_ = false;
        return;
    }

    // material.type
    if (!mat["type"]) {
        error_msg_ = "Missing required parameter: material.type";
        valid_ = false;
        return;
    }
    auto mat_type_str = mat["type"].as<std::string>();
    auto mat_type = StringToMaterialType(mat_type_str); // validate - will be abort if invalid type

    // Coupled (fluid-solid) materials validate nested fluid/solid blocks
    // instead of a flat format+params block. Short-circuit the rest of the
    // flat-format validation on success.
    if (mat_type == MaterialType::Coupled) {
        if (!ValidateMaterialSubNode(mat["fluid"], "fluid", error_msg_)) {
            valid_ = false;
            return;
        }
        if (!ValidateMaterialSubNode(mat["solid"], "solid", error_msg_)) {
            valid_ = false;
            return;
        }
        const int fa = mat["fluid"]["attribute"].as<int>();
        const int sa = mat["solid"]["attribute"].as<int>();
        if (fa == sa) {
            error_msg_ = "material.fluid.attribute and material.solid.attribute "
                         "must differ (got " + std::to_string(fa) + " for both)";
            valid_ = false;
            return;
        }
        // Physical consistency: fluid must be acoustic, solid must be elastic.
        const auto ft = StringToMaterialType(mat["fluid"]["type"].as<std::string>());
        const auto st = StringToMaterialType(mat["solid"]["type"].as<std::string>());
        if (ft != MaterialType::IsotropicAcoustic) {
            error_msg_ = "material.fluid.type must be acoustic (got "
                       + mat["fluid"]["type"].as<std::string>() + ")";
            valid_ = false;
            return;
        }
        if (st != MaterialType::IsotropicElastic &&
            st != MaterialType::AnisotropicElastic) {
            error_msg_ = "material.solid.type must be elastic (got "
                       + mat["solid"]["type"].as<std::string>() + ")";
            valid_ = false;
            return;
        }
        // No flat format/attenuation checks apply when type=coupled.
    } else {

    // material.format
    if (!mat["format"]) {
        error_msg_ = "Missing required parameter: material.format";
        valid_ = false;
        return;
    }
    std::string mat_format = mat["format"].as<std::string>();

    // Validate based on material format
    if (mat_format == "constant") {
        
        if (mat_type == MaterialType::IsotropicElastic) {
            if (!mat["vp"] || !mat["vs"] || !mat["rho"]) {
                error_msg_ = "Constant elastic material requires vp, vs, and rho parameters";
                valid_ = false;
                return;
            }
        } else if (mat_type == MaterialType::IsotropicAcoustic) {
            if (!mat["vp"] || !mat["rho"]) {
                error_msg_ = "Constant acoustic material requires vp and rho parameters";
                valid_ = false;
                return;
            }
        } else {
            error_msg_ = "Constant format not supported for anisotropic materials";
            valid_ = false;
            return;
        }

    } else if (mat_format == "hdf5") {

        MFEM_ABORT("HDF5 material format validation not yet implemented.");

    } else if (mat_format == "grid") {
        if (!mat["file"]) {
            error_msg_ = "Grid material format requires 'file' parameter";
            valid_ = false;
            return;
        }
    } else if (mat_format == "by_attribute") {
        if (!mat["file"]) {
            error_msg_ = "by_attribute material format requires 'file' parameter";
            valid_ = false;
            return;
        }
    } else if (mat_format == "by_attribute_mixed") {
        if (!mat["file"]) {
            error_msg_ = "by_attribute_mixed material format requires 'file' parameter";
            valid_ = false;
            return;
        }
    } else if (mat_format == "adios2") {
        if (!mat["vp_file"]) {
            error_msg_ = "ADIOS2 material format requires 'vp_file' parameter";
            valid_ = false;
            return;
        }
        if (!mat["rho_file"]) {
            error_msg_ = "ADIOS2 material format requires 'rho_file' parameter";
            valid_ = false;
            return;
        }
        if (mat_type == MaterialType::IsotropicElastic && !mat["vs_file"]) {
            error_msg_ = "ADIOS2 elastic material format requires 'vs_file' parameter";
            valid_ = false;
            return;
        }
    } else {
        error_msg_ = "Invalid material.format: " + mat_format;
        valid_ = false;
        return;
    }

    // Validate attenuation when enabled
    YAML::Node atten = mat["attenuation"];
    if (atten && !CheckKnownKeys(atten, {
            "enabled", "f0", "n_units",
            "Qkappa", "Qmu", "Qkappa_file", "Qmu_file"
        }, "material.attenuation", error_msg_)) {
        valid_ = false;
        return;
    }
    bool atten_enabled = atten && atten["enabled"] && atten["enabled"].as<bool>();
    if (atten_enabled) {
        // f0 is required
        if (!atten["f0"]) {
            error_msg_ = "Missing required parameter: material.attenuation.f0 (when attenuation enabled)";
            valid_ = false;
            return;
        }
        // n_units is required
        if (!atten["n_units"]) {
            error_msg_ = "Missing required parameter: material.attenuation.n_units (when attenuation enabled)";
            valid_ = false;
            return;
        }
        // Qkappa is required for constant format
        if (mat_format == "constant") {
            if (!atten["Qkappa"]) {
                error_msg_ = "Missing required parameter: material.attenuation.Qkappa (when attenuation enabled with constant format)";
                valid_ = false;
                return;
            }
            // Qmu required for elastic
            if (mat_type == MaterialType::IsotropicElastic) { //should be added for anisotropic, others.
                if (!atten["Qmu"]) {
                    error_msg_ = "Missing required parameter: material.attenuation.Qmu (when attenuation enabled with constant elastic format)";
                    valid_ = false;
                    return;
                }
            }
        }
    }

    }  // end else (flat material validation)

    //=============================================================================
    // Validate boundary Conditions Section
    //=============================================================================
    YAML::Node bc = root_["boundary"];
    if (!bc) {
        error_msg_ = "Missing required section: boundary";
        valid_ = false;
        return;
    }
    // Explicit migration guidance for the removed `free_surface:` block
    // (must precede the generic strict-key check so users see the
    // tailored message rather than the generic unknown-key one).
    if (bc["free_surface"]) {
        error_msg_ = "boundary.free_surface has been removed. For elastic "
                     "domains the free surface is the natural BC (simply "
                     "delete the block); for fluid domains use "
                     "`boundary.dirichlet.attributes: [<side|attr>]` "
                     "to enforce pressure = 0.";
        valid_ = false;
        return;
    }
    if (!CheckKnownKeys(bc,
            {"absorbing", "dirichlet"},
            "boundary", error_msg_)) {
        valid_ = false;
        return;
    }
    YAML::Node abc = bc["absorbing"];
    if (!abc) {
        error_msg_ = "Missing required section: boundary.absorbing";
        valid_ = false;
        return;
    }
    if (!CheckKnownKeys(abc,
            {"type", "sides", "thickness", "alpha"},
            "boundary.absorbing", error_msg_)) {
        valid_ = false;
        return;
    }
    if (!abc["type"]) {
        error_msg_ = "Missing required parameter: boundary.absorbing.type";
        valid_ = false;
        return;
    }
    std::string abc_type = abc["type"].as<std::string>();
    if (abc_type != "cerjan") {
        error_msg_ = "Invalid parameter: boundary.absorbing.type (must be 'cerjan')";
        valid_ = false;
        return;
    }
    //when dim == 2, sides should be bottom, top, left, right
    //when dim == 3, sides should be bottom, top, left, right, front, back
    if(!abc["sides"]) {
        error_msg_ = "Missing required parameter: boundary.absorbing.sides";
        valid_ = false;
        return;
    }
    std::vector<std::string> sides = abc["sides"].as<std::vector<std::string>>();
    bool noAbcSides = sides.empty();
    if(!noAbcSides){ 
        for (const auto& side : sides) {
            int parsed_side = ParseBoundarySide(side, dim);
            if (parsed_side < 0) {
                error_msg_ = "Invalid boundary side: " + side;
                valid_ = false;
                return;
            }
        }
    }

    if (!abc["thickness"]) {
        error_msg_ = "Missing required parameter: boundary.absorbing.thickness";
        valid_ = false;
        return;
    }
    if (!abc["alpha"]){
        error_msg_ = "Missing required parameter: boundary.absorbing.alpha";
        valid_ = false;
        return;
    }
    else{
        real_t alpha = abc["alpha"].as<real_t>();
        if (alpha < 0) {
            error_msg_ = "Invalid parameter: boundary .absorbing.alpha (must be non-negative)";
            valid_ = false;
            return;
        }
    }

    // Dirichlet BC section (optional).
    // (The removed `free_surface:` block is rejected at the top of the
    // boundary validator, alongside the strict-key check, so it cannot
    // reach this point.)
    YAML::Node dirichlet = bc["dirichlet"];
    if (dirichlet) {
        if (!CheckKnownKeys(dirichlet, {"attributes"},
                "boundary.dirichlet", error_msg_)) {
            valid_ = false;
            return;
        }
        if (dirichlet["attributes"]) {
            std::vector<std::string> attrs = dirichlet["attributes"].as<std::vector<std::string>>();
            for (const auto& attr : attrs) {
                int parsed = ParseBoundaryAttributeValue(attr, dim);
                if (parsed < 0) {
                    error_msg_ = "Invalid boundary attribute: " + attr;
                    valid_ = false;
                    return;
                }
            }
        }
    }


    //=============================================================================
    // Validate Sources Section
    //=============================================================================
    if (!root_["sources"]) {
        error_msg_ = "Missing required section: sources";
        valid_ = false;
        return;
    }
    YAML::Node sources = root_["sources"];
    if (!CheckKnownKeys(sources, {"mode", "file", "format", "shot_id", "list"},
            "sources", error_msg_)) {
        valid_ = false;
        return;
    }

    // sources.mode
    if (!sources["mode"]) {
        error_msg_ = "Missing required parameter: sources.mode";
        valid_ = false;
        return;
    }

    // Input format ("yaml" default, "hdf5" reads /shots/<shot_id>/sources/).
    std::string src_format = "yaml";
    if (sources["format"]) {
        src_format = sources["format"].as<std::string>();
        if (src_format != "yaml" && src_format != "hdf5") {
            error_msg_ = "Invalid parameter: sources.format='" + src_format
                       + "' (must be 'yaml' or 'hdf5')";
            valid_ = false;
            return;
        }
        if (src_format == "hdf5" && !sources["file"]) {
            error_msg_ = "sources.format=hdf5 requires sources.file";
            valid_ = false;
            return;
        }
        if (src_format == "hdf5" && sources["list"]) {
            error_msg_ = "sources.format=hdf5 conflicts with sources.list "
                         "(provide one or the other)";
            valid_ = false;
            return;
        }
    }
    if (sources["shot_id"]) {
        const int sid = sources["shot_id"].as<int>();
        if (sid < 0) {
            error_msg_ = "Invalid parameter: sources.shot_id must be >= 0";
            valid_ = false;
            return;
        }
    }

    // Validate inline sources if present
    YAML::Node src_list = sources["list"];

    if (sources["file"]) {
        //check file existence
        std::string src_file = sources["file"].as<std::string>();
        std::ifstream infile(src_file);
        if (!infile.good()) {
            error_msg_ = "Source file not found: " + src_file;
            valid_ = false;
            return;
        }
    }
    else if (src_format != "hdf5") {
        if (src_list && src_list.IsSequence()) {
            for (size_t i = 0; i < src_list.size(); i++) {
                YAML::Node src = src_list[i];
                std::string src_idx = "sources.list[" + std::to_string(i) + "]";

                if (!CheckKnownKeys(src, {
                        "id", "name", "type", "location", "direction",
                        "moment_tensor", "wavelet", "observed"
                    }, src_idx, error_msg_)) {
                    valid_ = false;
                    return;
                }

                // type required
                if (!src["type"]) {
                    error_msg_ = "Missing required parameter: " + src_idx + ".type";
                    valid_ = false;
                    return;
                }
                std::string src_type = src["type"].as<std::string>();
                if (src_type == "force") {
                    if(!src["direction"]) {
                        error_msg_ = "Missing required parameter: " + src_idx + ".direction (for force source)";
                        valid_ = false;
                        return;
                    }
                    std::vector<real_t> dir = src["direction"].as<std::vector<real_t>>();
                    if (dir.size() != dim) {
                        error_msg_ = "Invalid parameter: " + src_idx + ".direction (must have length equal to dimension)";
                        valid_ = false;
                        return;
                    }
                }
                else if (src_type == "moment_tensor") {
                    if (!src["moment_tensor"]) {
                        error_msg_ = "Missing required parameter: " + src_idx + ".moment_tensor (for moment_tensor source)";
                        valid_ = false;
                        return;
                    }
                    YAML::Node mt = src["moment_tensor"];
                    if (!CheckKnownKeys(mt,
                            {"Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"},
                            src_idx + ".moment_tensor", error_msg_)) {
                        valid_ = false;
                        return;
                    }
                    if (dim == 2) {
                        // In 2D, moment tensor source requires Mxx, Myy, Mxy
                        if (!mt["Mxx"] || !mt["Myy"] || !mt["Mxy"]) {
                            error_msg_ = "Missing required moment tensor components (Mxx, Myy, Mxy) for 2D moment_tensor source at " + src_idx;
                            valid_ = false;
                            return;
                        }
                    } else {
                        // In 3D, moment tensor source requires all 6 components
                        if (!mt["Mxx"] || !mt["Myy"] || !mt["Mzz"] ||
                            !mt["Mxy"] || !mt["Mxz"] || !mt["Myz"]) {
                            error_msg_ = "Missing required moment tensor components (Mxx, Myy, Mzz, Mxy, Mxz, Myz) for 3D moment_tensor source at " + src_idx;
                            valid_ = false;
                            return;
                        }
                    }
                }
                else if (src_type != "pressure") {
                    error_msg_ = "Invalid source.type at " + src_idx + " (must be 'force', 'moment_tensor', or 'pressure')";
                    valid_ = false;
                    return;
                }

                // location required
                if (!src["location"]) {
                    error_msg_ = "Missing required parameter: " + src_idx + ".location";
                    valid_ = false;
                    return;
                }
                std::vector<real_t> loc = src["location"].as<std::vector<real_t>>();
                if (loc.size() != dim) {
                    error_msg_ = "Invalid parameter: " + src_idx + ".location (must have length equal to dimension)";
                    valid_ = false;
                    return;
                }


                // wavelet section required
                if (!src["wavelet"]) {
                    error_msg_ = "Missing required section: " + src_idx + ".wavelet";
                    valid_ = false;
                    return;
                }
                YAML::Node wv = src["wavelet"];
                if (!CheckKnownKeys(wv,
                        {"type", "frequency", "amplitude", "delay", "file"},
                        src_idx + ".wavelet", error_msg_)) {
                    valid_ = false;
                    return;
                }

                // wavelet.type required
                if (!wv["type"]) {
                    error_msg_ = "Missing required parameter: " + src_idx + ".wavelet.type";
                    valid_ = false;
                    return;
                }

                std::string wv_type = wv["type"].as<std::string>();

                if (wv_type == "ricker" || wv_type == "gaussian") {
                    // frequency required for analytic wavelets (PPW check)
                    if (!wv["frequency"]) {
                        error_msg_ = "Missing required parameter: " + src_idx + ".wavelet.frequency";
                        valid_ = false;
                        return;
                    }
                    // wavelet.amplitude required
                    if (!wv["amplitude"]) {
                        error_msg_ = "Missing required parameter: " + src_idx + ".wavelet.amplitude";
                        valid_ = false;
                        return;
                    }
                    // wavelet.delay required
                    if (!wv["delay"]) {
                        error_msg_ = "Missing required parameter: " + src_idx + ".wavelet.delay";
                        valid_ = false;
                        return;
                    }
                }
                else if (wv_type == "external") {
                    // wavelet.file required
                    // amplitude and delay are NOT required for external
                    if (!wv["file"]) {
                        error_msg_ = "Missing required parameter: " + src_idx + ".wavelet.file";
                        valid_ = false;
                        return;
                    }
                    std::string wv_file = wv["file"].as<std::string>();
                    std::ifstream infile(wv_file);
                    if (!infile.good()) {
                        error_msg_ = "Wavelet file not found: " + wv_file + " (for external wavelet at " + src_idx + ")";
                        valid_ = false;
                        return;
                    }
                }
                else {
                    error_msg_ = "Invalid wavelet.type at " + src_idx + " (must be 'ricker', 'gaussian', or 'external')";
                    valid_ = false;
                    return;
                }
            }
        }
    }

    //=============================================================================
    // Validate Receivers Section
    // (optional for inversion mode — receiver positions come from observed SU files)
    //=============================================================================
    bool is_inversion_mode = sim["mode"] && (sim["mode"].as<std::string>() == "inversion"
                                             || sim["mode"].as<std::string>() == "misfit_only");
    if (!root_["receivers"] && !is_inversion_mode) {
        error_msg_ = "Missing required section: receivers";
        valid_ = false;
        return;
    }

    if (root_["receivers"]) {
    YAML::Node receivers = root_["receivers"];
    if (!CheckKnownKeys(receivers,
            {"type", "output", "file", "format", "shot_id", "list", "line"},
            "receivers", error_msg_)) {
        valid_ = false;
        return;
    }

    // Input format ("yaml" default, "hdf5" reads /shots/<shot_id>/receivers/).
    if (receivers["format"]) {
        const std::string in_fmt = receivers["format"].as<std::string>();
        if (in_fmt != "yaml" && in_fmt != "hdf5") {
            error_msg_ = "Invalid parameter: receivers.format='" + in_fmt
                       + "' (must be 'yaml' or 'hdf5')";
            valid_ = false;
            return;
        }
        if (in_fmt == "hdf5" && !receivers["file"]) {
            error_msg_ = "receivers.format=hdf5 requires receivers.file";
            valid_ = false;
            return;
        }
    }
    if (receivers["shot_id"]) {
        const int sid = receivers["shot_id"].as<int>();
        if (sid < 0) {
            error_msg_ = "Invalid parameter: receivers.shot_id must be >= 0";
            valid_ = false;
            return;
        }
    }

    // receivers.type required (parent-level only)
    if (!receivers["type"]) {
        error_msg_ = "Missing required parameter: receivers.type";
        valid_ = false;
        return;
    }

    std::vector<std::string> types = receivers["type"].as<std::vector<std::string>>();
    for (const auto& type_str : types) {
        auto rec_type = StringToReceiverType(type_str); // validate - will be abort if invalid type
    }

    // receivers.output section
    if (!receivers["output"]) {
        error_msg_ = "Missing required section: receivers.output";
        valid_ = false;
        return;
    }
    YAML::Node recv_output = receivers["output"];
    // Tailored migration message for the removed singular `format:` key
    // (precedes strict-key so the user sees it rather than the generic
    // unknown-key message).
    if (recv_output["format"]) {
        error_msg_ = "receivers.output.format has been removed. "
                     "Use `receivers.output.formats: [{type: <name>}, ...]` "
                     "(mapping list).";
        valid_ = false;
        return;
    }
    if (!CheckKnownKeys(recv_output, {"formats", "filename"},
            "receivers.output", error_msg_)) {
        valid_ = false;
        return;
    }
    if (!recv_output["formats"]) {
        error_msg_ = "Missing required parameter: receivers.output.formats";
        valid_ = false;
        return;
    }

    // Canonical form: `formats: [{type: ascii}, {type: su}, ...]`
    YAML::Node fmt_node = recv_output["formats"];
    std::vector<std::string> recv_formats;
    if (!fmt_node.IsSequence() || fmt_node.size() == 0) {
        error_msg_ = "Invalid parameter: receivers.output.formats must be a "
                     "non-empty list of mappings (`[{type: <name>}, ...]`).";
        valid_ = false;
        return;
    }
    for (const auto& item : fmt_node) {
        if (!item.IsMap() || !item["type"]) {
            error_msg_ = "Invalid parameter: receivers.output.formats "
                         "entries must be mappings with a `type:` key.";
            valid_ = false;
            return;
        }
        recv_formats.push_back(item["type"].as<std::string>());
    }

    bool needs_filename = false;
    std::set<std::string> seen;
    for (const auto& f : recv_formats) {
        if (f != "ascii" && f != "hdf5" && f != "su") {
            error_msg_ = "Invalid parameter: receivers.output.format entry '" + f +
                         "' (must be 'ascii', 'hdf5', or 'su')";
            valid_ = false;
            return;
        }
        if (!seen.insert(f).second) {
            error_msg_ = "Invalid parameter: receivers.output.format (duplicate entry '" + f + "')";
            valid_ = false;
            return;
        }
        if (f == "hdf5" || f == "su") needs_filename = true;
    }
    if (needs_filename && !recv_output["filename"]) {
        error_msg_ = "Missing required parameter: receivers.output.filename (required for hdf5/su)";
        valid_ = false;
        return;
    }





    // Skip inline list/line validation when external file is specified
    // (external file content is validated at parse time in ParseReceivers)
    bool has_external_file = receivers["file"] && !receivers["file"].as<std::string>().empty();
    if (!has_external_file) {
        // Validate receiver line if present
        YAML::Node line = receivers["line"];
        if (line) {
            if (!CheckKnownKeys(line,
                    {"start", "end", "count", "prefix"},
                    "receivers.line", error_msg_)) {
                valid_ = false;
                return;
            }
            if (!line["start"]) {
                error_msg_ = "Missing required parameter: receivers.line.start";
                valid_ = false;
                return;
            }
            if (!line["end"]) {
                error_msg_ = "Missing required parameter: receivers.line.end";
                valid_ = false;
                return;
            }
            if (!line["count"]) {
                error_msg_ = "Missing required parameter: receivers.line.count";
                valid_ = false;
                return;
            }
            if (!line["prefix"]) {
                error_msg_ = "Missing required parameter: receivers.line.prefix";
                valid_ = false;
                return;
            }
        }

        // Validate inline receivers if present
        YAML::Node recv_list = receivers["list"];
        if (recv_list && recv_list.IsSequence()) {
            for (size_t i = 0; i < recv_list.size(); i++) {
                YAML::Node rec = recv_list[i];
                std::string rec_idx = "receivers.list[" + std::to_string(i) + "]";

                if (!CheckKnownKeys(rec,
                        {"name", "location", "type", "weight"},
                        rec_idx, error_msg_)) {
                    valid_ = false;
                    return;
                }

                // location required
                if (!rec["location"]) {
                    error_msg_ = "Missing required parameter: " + rec_idx + ".location";
                    valid_ = false;
                    return;
                }

                std::vector<real_t> loc = rec["location"].as<std::vector<real_t>>();
                if (loc.size() != dim) {
                    error_msg_ = "Invalid parameter: " + rec_idx + ".location (must have length equal to dimension)";
                    valid_ = false;
                    return;
                }

                //name required
                if (!rec["name"]) {
                    error_msg_ = "Missing required parameter: " + rec_idx + ".name";
                    valid_ = false;
                    return;
                }

            }
        }
    }
    }  // if (root_["receivers"])

    //=============================================================================
    // Validate Device Section
    //=============================================================================
    if (!root_["device"].IsDefined()) {
        error_msg_ = "Missing required section: device";
        valid_ = false;
        return;
    }

    YAML::Node device_node = root_["device"];
    if (!CheckKnownKeys(device_node,
            {"type", "seismo_buffer_steps"},
            "device", error_msg_)) {
        valid_ = false;
        return;
    }
    if (!device_node["type"]) {
        error_msg_ = "Missing required parameter: device.type";
        valid_ = false;
        return;
    }

    //=============================================================================
    // Validate Simulation Mode and Inversion Parameters
    //=============================================================================
    std::string sim_mode = "forward";  // default
    if (sim["mode"]) {
        sim_mode = sim["mode"].as<std::string>();
        if (sim_mode != "forward" && sim_mode != "inversion"
            && sim_mode != "misfit_only") {
            error_msg_ = "Invalid simulation.mode: " + sim_mode
                         + " (must be 'forward', 'inversion', or 'misfit_only')";
            valid_ = false;
            return;
        }
    }

    if (sim_mode == "inversion" || sim_mode == "misfit_only") {
        // misfit_type: optional, validate if present
        if (sim["misfit_type"]) {
            std::string mt = sim["misfit_type"].as<std::string>();
            if (mt != "l2_waveform" && mt != "normalized_correlation") {
                error_msg_ = "Invalid simulation.misfit_type: " + mt
                             + " (must be 'l2_waveform' or 'normalized_correlation')";
                valid_ = false;
                return;
            }
        }
        // num_checkpoints: optional, default 10
        if (sim["num_checkpoints"]) {
            int ncp = sim["num_checkpoints"].as<int>();
            if (ncp < 1) {
                error_msg_ = "Invalid simulation.num_checkpoints:"
                             " must be >= 1";
                valid_ = false;
                return;
            }
        }
        // checkpoint_storage: optional, validate if present
        if (sim["checkpoint_storage"]) {
            std::string cs = sim["checkpoint_storage"].as<std::string>();
            if (cs != "memory" && cs != "disk") {
                error_msg_ = "Invalid simulation.checkpoint_storage:"
                             " must be 'memory' or 'disk'";
                valid_ = false;
                return;
            }
            if (cs == "disk" && !sim["checkpoint_dir"]) {
                error_msg_ = "Missing required parameter:"
                             " simulation.checkpoint_dir"
                             " (when checkpoint_storage is 'disk')";
                valid_ = false;
                return;
            }
        }
        // checkpoint_device: optional, validate if present
        if (sim["checkpoint_device"]) {
            std::string cd = sim["checkpoint_device"].as<std::string>();
            if (cd != "auto" && cd != "host" && cd != "device") {
                error_msg_ = "Invalid simulation.checkpoint_device:"
                             " must be 'auto', 'host', or 'device'";
                valid_ = false;
                return;
            }
        }

        // Validate per-source observed data
        if (src_list && src_list.IsSequence()) {
            for (size_t i = 0; i < src_list.size(); i++) {
                YAML::Node src = src_list[i];
                std::string src_idx = "sources.list[" + std::to_string(i) + "]";

                if (!src["observed"]) {
                    error_msg_ = "Missing required section: " + src_idx
                                 + ".observed (when mode is 'inversion')";
                    valid_ = false;
                    return;
                }
                YAML::Node obs = src["observed"];

                // Detect legacy SU schema and emit migration hint.
                if (obs["format"] || obs["files"]) {
                    error_msg_ = src_idx + ".observed uses the legacy SU "
                                 "schema (format/files). SEMSWS accepts "
                                 "only the canonical HDF5 form: "
                                 "observed: { file: <path.h5>, resample: { "
                                 "enabled: false, method: lanczos, lanczos_a: 4 } }.";
                    valid_ = false;
                    return;
                }

                if (!CheckKnownKeys(obs, {"file", "resample"},
                        src_idx + ".observed", error_msg_)) {
                    valid_ = false;
                    return;
                }

                if (!obs["file"]) {
                    error_msg_ = "Missing required parameter: " + src_idx
                                 + ".observed.file (path to HDF5 file)";
                    valid_ = false;
                    return;
                }

                if (obs["resample"]) {
                    YAML::Node rs = obs["resample"];
                    if (!CheckKnownKeys(rs,
                            {"enabled", "method", "lanczos_a"},
                            src_idx + ".observed.resample", error_msg_)) {
                        valid_ = false;
                        return;
                    }
                    if (rs["method"]) {
                        std::string m = rs["method"].as<std::string>();
                        if (m != "lanczos" && m != "linear") {
                            error_msg_ = "Invalid " + src_idx
                                + ".observed.resample.method: " + m
                                + " (must be 'lanczos' or 'linear')";
                            valid_ = false;
                            return;
                        }
                    }
                }
            }
        }
    }
}

// =============================================================================
// Helper Methods
// =============================================================================

std::string YamlConfig::ReadString(const YAML::Node& node, const std::string& key) const {
    if (!node || !node[key]) {
        MFEM_ABORT("Required string parameter '" << key << "' not found in config");
    }
    return node[key].as<std::string>();
}

real_t YamlConfig::Readreal_t(const YAML::Node& node, const std::string& key) const {
    if (!node || !node[key]) {
        MFEM_ABORT("Required real parameter '" << key << "' not found in config");
    }
    return node[key].as<real_t>();
}

int YamlConfig::ReadInt(const YAML::Node& node, const std::string& key) const {
    if (!node || !node[key]) {
        MFEM_ABORT("Required int parameter '" << key << "' not found in config");
    }
    return node[key].as<int>();
}

bool YamlConfig::ReadBool(const YAML::Node& node, const std::string& key) const {
    if (!node || !node[key]) {
        MFEM_ABORT("Required bool parameter '" << key << "' not found in config");
    }
    return node[key].as<bool>();
}

void YamlConfig::ReadRealArray(const YAML::Node& node, const std::string& key,
                                real_t* arr, int size) const {
    if (!node || !node[key] || !node[key].IsSequence()) {
        MFEM_ABORT("Required real array parameter '" << key << "' not found in config");
    }
    int n = std::min(size, static_cast<int>(node[key].size()));
    for (int i = 0; i < n; i++) {
        arr[i] = static_cast<real_t>(node[key][i].as<real_t>());
    }
    // Zero-fill remaining elements
    for (int i = n; i < size; i++) {
        arr[i] = 0.0;
    }
}

void YamlConfig::ReadIntArray(const YAML::Node& node, const std::string& key,
                               int* arr, int size) const {
    if (!node || !node[key] || !node[key].IsSequence()) {
        MFEM_ABORT("Required int array parameter '" << key << "' not found in config");
    }
    int n = std::min(size, static_cast<int>(node[key].size()));
    for (int i = 0; i < n; i++) {
        arr[i] = node[key][i].as<int>();
    }
    // Zero-fill remaining elements
    for (int i = n; i < size; i++) {
        arr[i] = 0;
    }
}

std::vector<std::string> YamlConfig::ReadStringArray(const YAML::Node& node,
                                                      const std::string& key) const {
    std::vector<std::string> result;
    if (node && node[key] && node[key].IsSequence()) {
        for (size_t i = 0; i < node[key].size(); i++) {
            result.push_back(node[key][i].as<std::string>());
        }
    }
    return result;
}

// =============================================================================
// Simulation Section
// =============================================================================

std::string YamlConfig::GetName() const {
    // Name is at root level, with default set in Validate()
    if (root_["name"]) {
        return root_["name"].as<std::string>();
    }
    return "Seismic wave simulation";
}

// GetPhysics() removed - physics type is inferred from MaterialType
// GetBackend() removed - unused

int YamlConfig::GetDimension() const {
    // Required - validated in Validate()
    return root_["simulation"]["dimension"].as<int>();
}

int YamlConfig::GetOrder() const {
    // order is required (validated in Validate())
    YAML::Node sim = root_["simulation"];
    // if (!sim || !sim["order"]) {
    //     MFEM_ABORT("simulation.order is required but not found in config");
    // }
    return sim["order"].as<int>();
}

int YamlConfig::GetNumSteps() const {
    // steps is required (validated in Validate())
    YAML::Node time_node = root_["simulation"]["time"];
    // if (!time_node || !time_node["steps"]) {
    //     MFEM_ABORT("simulation.time.steps is required but not found in config");
    // }
    return time_node["steps"].as<int>();
}

real_t YamlConfig::GetDt() const {
    // dt is required (validated in Validate()).
    return root_["simulation"]["time"]["dt"].as<real_t>();
}

real_t YamlConfig::GetT0() const {
    YAML::Node time_node = root_["simulation"]["time"];
    if (time_node && time_node["t0"]) {
        return time_node["t0"].as<real_t>();
    }
    return 0.0;
}

real_t YamlConfig::GetCflFactor() const {
    // Required - validated in Validate()
    return root_["simulation"]["time"]["cfl_factor"].as<real_t>();
}

std::string YamlConfig::GetOutputDirectory() const {
    // Required - validated in Validate()
    return root_["simulation"]["output"]["directory"].as<std::string>();
}

// GetSeismogramFormat() removed - unused

bool YamlConfig::IsWavefieldOutputEnabled() const {
    if (!root_["simulation"] || !root_["simulation"]["output"] ||
        !root_["simulation"]["output"]["wavefield"] ||
        !root_["simulation"]["output"]["wavefield"]["enabled"]) {
        return false;
    }
    return root_["simulation"]["output"]["wavefield"]["enabled"].as<bool>();
}

int YamlConfig::GetWavefieldInterval() const {
    // Required when wavefield enabled - validated in Validate()
    // Caller should check IsWavefieldOutputEnabled() first
    YAML::Node wf_node = root_["simulation"]["output"]["wavefield"];
    return wf_node["interval"].as<int>();
}

// NOTE: GetWavefieldFormat() / WavefieldFormat() have been removed.
// Use GetWavefieldOutputConfig() and iterate its `formats` list.

// =============================================================================
// Multi-format visualization output configuration
// =============================================================================

namespace {

/// Parse a single output format entry from YAML.
///
/// Accepts two syntaxes per list entry:
///   * Mapping  (full):  `{type: "gmt", resolution: [100, 100], …}`
///   * Scalar   (short): `"gmt"`  — equivalent to `{type: "gmt"}` with
///                       all other options at defaults. Useful when
///                       only the format name matters (common for
///                       material.formats).
OutputFormatConfig ParseOutputFormat(const YAML::Node& node) {
    // Canonical form: a mapping with at least a `type:` key, e.g.
    //   { type: "gmt", resolution: [200, 100], components: [-1, 0] }
    // Scalar shorthand (`formats: ["gmt"]`) and missing-`type:` are
    // explicit errors.
    OutputFormatConfig fmt;
    if (!node.IsMap()) {
        MFEM_ABORT("output format entry must be a mapping like "
                   "`{type: <name>, ...}`; bare-string shorthand "
                   "(e.g., `formats: [\"gmt\"]`) has been removed.");
    }
    if (!node["type"]) {
        MFEM_ABORT("output format entry is missing the required `type:` key.");
    }
    // Strict-key validation on format entries — catches renamed or
    // mistyped keys that would otherwise be silently ignored.
    RequireKnownKeys(node, {
        "type",
        "refinement", "data_format", "compression",
        "resolution", "components", "cross_sections"
    }, "output format entry");
    fmt.type = node["type"].as<std::string>();

    // ParaView options
    if (node["refinement"]) {
        fmt.refinement = node["refinement"].as<int>();
    }
    if (node["data_format"]) {
        fmt.data_format = node["data_format"].as<std::string>();
    }
    if (node["compression"]) {
        fmt.compression = node["compression"].as<int>();
    }

    // GMT options
    if (node["resolution"]) {
        auto res = node["resolution"];
        if (res.IsSequence() && res.size() == 2) {
            fmt.resolution[0] = res[0].as<int>();
            fmt.resolution[1] = res[1].as<int>();
        }
    }
    if (node["components"]) {
        fmt.components.clear();
        for (size_t i = 0; i < node["components"].size(); ++i) {
            fmt.components.push_back(node["components"][i].as<int>());
        }
    }
    if (node["cross_sections"]) {
        auto cs = node["cross_sections"];
        if (cs["yz"]) {
            for (size_t i = 0; i < cs["yz"].size(); ++i) {
                fmt.cross_sections.yz.push_back(cs["yz"][i].as<real_t>());
            }
        }
        if (cs["xz"]) {
            for (size_t i = 0; i < cs["xz"].size(); ++i) {
                fmt.cross_sections.xz.push_back(cs["xz"][i].as<real_t>());
            }
        }
        if (cs["xy"]) {
            for (size_t i = 0; i < cs["xy"].size(); ++i) {
                fmt.cross_sections.xy.push_back(cs["xy"][i].as<real_t>());
            }
        }
    }

    return fmt;
}

}  // anonymous namespace

namespace {

/// Apply an output.wavefield(-like) YAML node's keys onto an existing
/// WavefieldOutputConfig, overriding whichever fields are present in
/// the node and leaving the rest untouched. Shared between the
/// top-level accessor and the per-side override accessor so both paths
/// follow identical parsing rules.
void ApplyWavefieldNode(const YAML::Node& wf, WavefieldOutputConfig& config) {
    if (wf["enabled"]) {
        config.enabled = wf["enabled"].as<bool>();
    }
    if (wf["interval"]) {
        config.interval = wf["interval"].as<int>();
    }
    if (wf["fields"]) {
        config.fields.clear();
        static const std::vector<std::string> valid_wf_fields =
            {"DISP", "VEL", "ACC", "PS"};
        for (size_t i = 0; i < wf["fields"].size(); ++i) {
            std::string f = wf["fields"][i].as<std::string>();
            if (std::find(valid_wf_fields.begin(), valid_wf_fields.end(), f)
                == valid_wf_fields.end()) {
                MFEM_ABORT("Unknown wavefield field '" + f
                           + "' (valid: DISP, VEL, ACC, PS)");
            }
            config.fields.push_back(f);
        }
    }
    if (wf["formats"]) {
        // Canonical form — replaces any previously accumulated formats
        // (per-side override replaces the inherited base).
        config.formats.clear();
        for (size_t i = 0; i < wf["formats"].size(); ++i) {
            config.formats.push_back(ParseOutputFormat(wf["formats"][i]));
        }
    }

    // Block-level `components:` acts as a broadcast override — useful
    // under a per-domain `fluid:` / `solid:` block where the user wants
    // (say) `components: [-1, 1, 2]` applied to every format on the
    // solid side without repeating the key inside each format entry.
    // Precedence: if specified at block level, it overrides whatever
    // was parsed per-format (the per-format default is `{-1}`, so the
    // typical case has nothing explicit to preserve anyway).
    if (wf["components"]) {
        std::vector<int> comps;
        for (size_t i = 0; i < wf["components"].size(); ++i) {
            comps.push_back(wf["components"][i].as<int>());
        }
        for (auto& fmt : config.formats) {
            fmt.components = comps;
        }
    }
}

}  // anonymous namespace

WavefieldOutputConfig YamlConfig::GetWavefieldOutputConfig() const {
    WavefieldOutputConfig config;

    if (!root_["simulation"] || !root_["simulation"]["output"] ||
        !root_["simulation"]["output"]["wavefield"]) {
        return config;
    }

    YAML::Node wf = root_["simulation"]["output"]["wavefield"];
    ApplyWavefieldNode(wf, config);
    if (!config.enabled) {
        // Drop any formats we may have parsed before enabled=false was
        // applied, to preserve the pre-refactor early-return behavior.
        config.formats.clear();
        config.fields.clear();
    }
    return config;
}

WavefieldOutputConfig YamlConfig::GetWavefieldOutputConfig(
    const std::string& side) const {
    // Start from the top-level defaults (inherits interval / fields /
    // formats). A per-side block under
    // `simulation.output.wavefield.<side>` then overrides whatever keys
    // it sets; unspecified keys stay as inherited.
    WavefieldOutputConfig config = GetWavefieldOutputConfig();

    if (!config.enabled) return config;
    if (!root_["simulation"] || !root_["simulation"]["output"] ||
        !root_["simulation"]["output"]["wavefield"]) {
        return config;
    }
    YAML::Node wf = root_["simulation"]["output"]["wavefield"];
    if (!wf[side]) return config;

    YAML::Node side_node = wf[side];
    ApplyWavefieldNode(side_node, config);

    // A side-level `enabled: false` explicitly mutes one domain's
    // wavefield output even when the global toggle is on.
    if (!config.enabled) {
        config.formats.clear();
        config.fields.clear();
    }
    return config;
}

namespace {

/// Shared parser for an `output.material`-like node. Mirrors
/// `ApplyWavefieldNode` — any key set in `mat` overrides the matching
/// field in `config`; unset keys leave `config` untouched. Used by both
/// the top-level accessor and the per-side override accessor so the
/// two paths follow identical rules.
void ApplyMaterialNode(const YAML::Node& mat, MaterialOutputConfig& config) {
    if (mat["enabled"]) {
        config.enabled = mat["enabled"].as<bool>();
    }
    if (mat["fields"]) {
        config.fields.clear();
        static const std::vector<std::string> valid_mat_fields =
            {"vp", "vs", "rho", "qkappa", "qmu"};
        for (size_t i = 0; i < mat["fields"].size(); ++i) {
            std::string f = mat["fields"][i].as<std::string>();
            if (std::find(valid_mat_fields.begin(), valid_mat_fields.end(), f)
                == valid_mat_fields.end()) {
                MFEM_ABORT("Unknown material field '" + f
                           + "' (valid: vp, vs, rho, qkappa, qmu)");
            }
            config.fields.push_back(f);
        }
    }
    if (mat["formats"]) {
        // Explicit formats list — replaces any previously accumulated
        // formats (per-side override should not inherit base formats
        // that the user is explicitly replacing).
        config.formats.clear();
        for (size_t i = 0; i < mat["formats"].size(); ++i) {
            config.formats.push_back(ParseOutputFormat(mat["formats"][i]));
        }
    }
}

}  // anonymous namespace

MaterialOutputConfig YamlConfig::GetMaterialOutputConfig() const {
    MaterialOutputConfig config;

    if (!root_["simulation"] || !root_["simulation"]["output"] ||
        !root_["simulation"]["output"]["material"]) {
        return config;
    }

    YAML::Node mat = root_["simulation"]["output"]["material"];
    ApplyMaterialNode(mat, config);
    if (!config.enabled) {
        config.formats.clear();
        config.fields.clear();
    }
    return config;
}

MaterialOutputConfig YamlConfig::GetMaterialOutputConfig(
    const std::string& side) const {
    // Start from the top-level defaults (inherits fields / formats).
    // A per-side block under `simulation.output.material.<side>` then
    // overrides whatever keys it sets; unspecified keys stay as
    // inherited. If the side block has `enabled: false` that side's
    // material output is muted even when the global toggle is on.
    MaterialOutputConfig config = GetMaterialOutputConfig();

    if (!config.enabled) return config;
    if (!root_["simulation"] || !root_["simulation"]["output"] ||
        !root_["simulation"]["output"]["material"]) {
        return config;
    }
    YAML::Node mat = root_["simulation"]["output"]["material"];
    if (!mat[side]) return config;

    ApplyMaterialNode(mat[side], config);
    if (!config.enabled) {
        config.formats.clear();
        config.fields.clear();
    }
    return config;
}

bool YamlConfig::GetMeshSave() const {
    if (!root_["mesh"] || !root_["mesh"]["save"]) {
        return false;
    }
    return root_["mesh"]["save"].as<bool>();
}

int YamlConfig::GetLogInterval() const {
    // Required - validated in Validate()
    return root_["simulation"]["output"]["log_interval"].as<int>();
}

std::string YamlConfig::GetSummaryFile() const {
    // Optional - returns empty string if not specified
    YAML::Node output_node = root_["simulation"]["output"];
    if (output_node && output_node["summary_file"]) {
        return output_node["summary_file"].as<std::string>();
    }
    return "";
}

// =============================================================================
// Mesh Section
// =============================================================================

real_t YamlConfig::GetMaxFreq() const {
    // Required - validated in Validate()
    return root_["mesh"]["max_freq"].as<real_t>();
}

real_t YamlConfig::GetPPW() const {
    // Required - validated in Validate()
    return root_["mesh"]["ppw"].as<real_t>();
}

std::string YamlConfig::GetMeshType() const {
    // Required - validated in Validate()
    return root_["mesh"]["type"].as<std::string>();
}

std::string YamlConfig::GetMeshFile() const {
    // Required for external mesh - validated in Validate()
    return root_["mesh"]["file"].as<std::string>();
}

std::string YamlConfig::GetMeshFormat() const {
    // Required for external mesh - validated in Validate()
    if (root_["mesh"]["format"]) {
        return root_["mesh"]["format"].as<std::string>();
    }
    return "";
}

void YamlConfig::GetMeshOrigin(real_t* origin) const {
    // Required for internal mesh - validated in Validate()
    int dim = GetDimension();
    YAML::Node orig = root_["mesh"]["origin"];
    for (int i = 0; i < dim; i++) {
        origin[i] = orig[i].as<real_t>();
    }
}

void YamlConfig::GetMeshSize(real_t* size) const {
    // Required for internal mesh - validated in Validate()
    int dim = GetDimension();
    YAML::Node sz = root_["mesh"]["size"];
    for (int i = 0; i < dim; i++) {
        size[i] = sz[i].as<real_t>();
    }
}

void YamlConfig::GetMeshElements(int* nel) const {
    // Required for internal mesh - validated in Validate()
    int dim = GetDimension();
    YAML::Node el = root_["mesh"]["elements"];
    for (int i = 0; i < dim; i++) {
        nel[i] = el[i].as<int>();
    }
}

real_t YamlConfig::GetMeshAttrYThreshold() const {
    YAML::Node m = root_["mesh"];
    if (m && m["attr_y_threshold"]) {
        return m["attr_y_threshold"].as<real_t>();
    }
    return std::numeric_limits<real_t>::quiet_NaN();
}

std::string YamlConfig::GetMeshPartition() const {
    // Optional - defaults to "metis"
    YAML::Node mesh = root_["mesh"];
    if (mesh["partition"]) {
        return mesh["partition"].as<std::string>();
    }
    return "metis";
}

void YamlConfig::GetPartitionGrid(int* nxyz) const {
    // Required when partition: cartesian
    int dim = GetDimension();
    YAML::Node grid = root_["mesh"]["partition_grid"];
    for (int i = 0; i < dim; i++) {
        nxyz[i] = grid[i].as<int>();
    }
}

// -----------------------------------------------------------------------------
// Partitioned Mesh (pre-partitioned files)
// -----------------------------------------------------------------------------

bool YamlConfig::UsePartitionedMesh() const {
    std::string mesh_type = GetMeshType();
    return (mesh_type == "partitioned");
}

std::string YamlConfig::GetPartitionDirectory() const {
    // Required when mesh.type == "partitioned"
    YAML::Node partitioned = root_["mesh"]["partitioned"];
    if (!partitioned || !partitioned["directory"]) {
        MFEM_ABORT("mesh.partitioned.directory is required when mesh.type is 'partitioned'");
    }
    return partitioned["directory"].as<std::string>();
}

int YamlConfig::GetPartitionCount() const {
    // Required when mesh.type == "partitioned"
    YAML::Node partitioned = root_["mesh"]["partitioned"];
    if (!partitioned || !partitioned["nparts"]) {
        MFEM_ABORT("mesh.partitioned.nparts is required when mesh.type is 'partitioned'");
    }
    return partitioned["nparts"].as<int>();
}

// =============================================================================
// Material Section
// =============================================================================

std::string YamlConfig::GetMaterialType() const {
    // Required - validated in Validate()
    auto mat = root_["material"]["type"].as<std::string>();
    StringToMaterialType(mat); // validate - will MFEM_ABORT if invalid
    return mat;
}

std::string YamlConfig::GetMaterialFormat() const {
    // Required - validated in Validate()
    return root_["material"]["format"].as<std::string>();
}

bool YamlConfig::IsCoupledMaterial() const {
    if (!root_["material"] || !root_["material"]["type"]) return false;
    return root_["material"]["type"].as<std::string>() == "coupled";
}

CoupledMaterialConfig YamlConfig::GetCoupledMaterialConfig() const {
    MFEM_VERIFY(IsCoupledMaterial(),
                "GetCoupledMaterialConfig called on a non-coupled material "
                "(material.type must be 'coupled')");
    CoupledMaterialConfig cc;
    YAML::Node mat = root_["material"];
    cc.fluid_attribute = mat["fluid"]["attribute"].as<int>();
    cc.solid_attribute = mat["solid"]["attribute"].as<int>();
    cc.fluid = ParseMaterialSubNode(mat["fluid"], "fluid");
    cc.solid = ParseMaterialSubNode(mat["solid"], "solid");
    return cc;
}

std::string YamlConfig::GetMaterialFile() const {
    // Required for hdf5/ascii format - validated in Validate()
    return root_["material"]["file"].as<std::string>();
}



void YamlConfig::GetConstantMaterialVpVsRho(real_t* vp, real_t* vs, real_t* rho) const {
    // Required for constant format - validated in Validate()
    YAML::Node mat = root_["material"];
    *vp = mat["vp"].as<real_t>();
    *vs = mat["vs"].as<real_t>();
    *rho = mat["rho"].as<real_t>();
}

void YamlConfig::GetConstantMaterialVpRho(real_t* vp, real_t* rho) const {
    // Required for constant acoustic format - validated in Validate()
    YAML::Node mat = root_["material"];
    *vp = mat["vp"].as<real_t>();
    *rho = mat["rho"].as<real_t>();
}

bool YamlConfig::IsAttenuationEnabled() const {
    YAML::Node atten = root_["material"]["attenuation"];
    if (!atten || !atten["enabled"]) {
        return false;
    }
    return atten["enabled"].as<bool>();
}

real_t YamlConfig::GetAttenuationF0() const {
    // Required when attenuation enabled - validated in Validate()
    return root_["material"]["attenuation"]["f0"].as<real_t>();
}

int YamlConfig::GetAttenuationNumUnits() const {
    // Required when attenuation enabled - validated in Validate()
    return root_["material"]["attenuation"]["n_units"].as<int>();
}

real_t YamlConfig::GetConstantQkappa() const {
    YAML::Node atten = root_["material"]["attenuation"];
    // If Qkappa_file is specified, return -1 to indicate "from file"
    if (atten && atten["Qkappa_file"]) {
        return -1.0;
    }
    // Required when attenuation enabled with constant format - validated in Validate()
    return atten["Qkappa"].as<real_t>();
}

real_t YamlConfig::GetConstantQmu() const {
    YAML::Node atten = root_["material"]["attenuation"];
    // If Qmu_file is specified, return -1 to indicate "from file"
    if (atten && atten["Qmu_file"]) {
        return -1.0;
    }
    // For acoustic materials, Qmu may not be specified - return -1 as sentinel
    if (!atten || !atten["Qmu"]) {
        return -1.0;
    }
    // Required when attenuation enabled with constant elastic format - validated in Validate()
    return atten["Qmu"].as<real_t>();
}

std::string YamlConfig::GetQkappaFile() const {
    // Optional - returns empty string if not specified (sentinel value OK)
    YAML::Node atten = root_["material"]["attenuation"];
    if (atten && atten["Qkappa_file"]) {
        return atten["Qkappa_file"].as<std::string>();
    }
    return "";
}

std::string YamlConfig::GetQmuFile() const {
    // Optional - returns empty string if not specified (sentinel value OK)
    YAML::Node atten = root_["material"]["attenuation"];
    if (atten && atten["Qmu_file"]) {
        return atten["Qmu_file"].as<std::string>();
    }
    return "";
}

// GetQkappaDataset(), GetQmuDataset() removed - unused
// GetAsciiMaterialParams() removed - unused
// HasMaterialByAttribute(), GetMaterialByAttribute() removed - unused

std::string YamlConfig::GetMaterialByAttributeFile() const {
    // Optional - returns empty string if not specified (sentinel value OK)
    YAML::Node mat = root_["material"];
    if (mat && mat["file"]) {
        return mat["file"].as<std::string>();
    }
    return "";
}

std::string YamlConfig::GetADIOS2VpFile() const {
    YAML::Node mat = root_["material"];
    if (mat && mat["vp_file"]) {
        return mat["vp_file"].as<std::string>();
    }
    return "";
}

std::string YamlConfig::GetADIOS2VsFile() const {
    YAML::Node mat = root_["material"];
    if (mat && mat["vs_file"]) {
        return mat["vs_file"].as<std::string>();
    }
    return "";
}

std::string YamlConfig::GetADIOS2RhoFile() const {
    YAML::Node mat = root_["material"];
    if (mat && mat["rho_file"]) {
        return mat["rho_file"].as<std::string>();
    }
    return "";
}

std::string YamlConfig::GetADIOS2QkappaFile() const {
    YAML::Node mat = root_["material"];
    if (mat && mat["qkappa_file"]) {
        return mat["qkappa_file"].as<std::string>();
    }
    return "";
}

std::string YamlConfig::GetADIOS2QmuFile() const {
    YAML::Node mat = root_["material"];
    if (mat && mat["qmu_file"]) {
        return mat["qmu_file"].as<std::string>();
    }
    return "";
}

bool YamlConfig::IsExportModelEnabled() const {
    YAML::Node mat = root_["material"];
    if (mat && mat["export_model"]) {
        return mat["export_model"].as<bool>();
    }
    return false;
}

std::string YamlConfig::GetExportModelDir() const {
    YAML::Node mat = root_["material"];
    if (mat && mat["export_dir"]) {
        return mat["export_dir"].as<std::string>();
    }
    return "./model/";
}

// =============================================================================
// Boundary Section
// =============================================================================

ABCConfig YamlConfig::GetABCConfig() const {
    ABCConfig config;
    YAML::Node abc = root_["boundary"]["absorbing"];

    if (!abc) {
        return config;  // No absorbing boundary defined
    }

    // type required when absorbing defined - validated in Validate()
    config.type = abc["type"].as<std::string>();
    config.sides = ReadStringArray(abc, "sides");
    // thickness required when absorbing defined - validated in Validate()
    config.thickness = abc["thickness"].as<real_t>();

    config.alpha = abc["alpha"].as<real_t>();

    return config;
}

std::vector<std::string> YamlConfig::GetDirichletAttributes() const {
    YAML::Node section = root_["boundary"]["dirichlet"];
    if (!section || !section["attributes"]) return {};
    return ReadStringArray(section, "attributes");
}

// =============================================================================
// Sources Section
// =============================================================================

std::string YamlConfig::GetSourceMode() const {
    // Required - validated in Validate()
    return root_["sources"]["mode"].as<std::string>();
}

std::string YamlConfig::GetSourceFile() const {
    // Optional - returns empty string if not specified (sentinel value OK)
    YAML::Node src = root_["sources"];
    if (src && src["file"]) {
        return src["file"].as<std::string>();
    }
    return "";
}

std::string YamlConfig::GetSourceFormat() const {
    YAML::Node src = root_["sources"];
    if (src && src["format"]) {
        return src["format"].as<std::string>();
    }
    return "yaml";
}

int YamlConfig::GetSourceShotId() const {
    YAML::Node src = root_["sources"];
    if (src && src["shot_id"]) {
        return src["shot_id"].as<int>();
    }
    return 0;
}

void YamlConfig::ParseSources() const {
    if (sources_parsed_) return;

    sources_cache_.clear();

    // HDF5 input branch: delegate to HDF5SourceReceiverReader and translate
    // each HDF5SourceEntry back into SourceDef so downstream
    // (LoadSourceConfig{2D,3D}) is unchanged.
    if (GetSourceFormat() == "hdf5") {
        const std::string h5_path = GetSourceFile();
        const int shot_id = GetSourceShotId();
        const int dim = GetDimension();
        HDF5SourceCatalog cat =
            HDF5SourceReceiverReader::ReadSources(h5_path, shot_id, dim);
        for (const auto& s : cat.sources) {
            SourceDef def;
            def.id = s.id;
            def.name = s.label.empty()
                ? "source_" + std::to_string(s.id)
                : s.label;
            def.type = s.type;
            for (int d = 0; d < 3; ++d) {
                def.location[d] = (d < static_cast<int>(s.position.size()))
                                      ? s.position[d] : real_t{0};
                def.direction[d] = (d < static_cast<int>(s.direction.size()))
                                       ? s.direction[d] : real_t{0};
            }
            // Moment tensor components in canonical order from the reader.
            // 3D fills M[0..5]; 2D fills M[0]=Mxx, M[1]=Myy, M[3]=Mxy
            // (matches the slot convention used by
            // ConfigLoaders.cpp::LoadSourceConfig2D).
            for (int i = 0; i < 6; ++i) def.M[i] = real_t{0};
            if (s.type == "moment_tensor") {
                if (s.moment_tensor.size() == 6) {
                    for (int i = 0; i < 6; ++i) def.M[i] = s.moment_tensor[i];
                } else if (s.moment_tensor.size() == 3) {
                    // 2D order from reader: {Mxx, Myy, Mxy}
                    def.M[0] = s.moment_tensor[0];
                    def.M[1] = s.moment_tensor[1];
                    def.M[3] = s.moment_tensor[2];
                }
            }
            // wavelet_type="hdf5" signals "use stf_samples directly".
            // frequency / amplitude / delay are unused; external_file empty.
            def.wavelet_type = "hdf5";
            def.frequency = real_t{0};
            def.amplitude = real_t{1};
            def.delay = real_t{0};
            def.stf_samples = s.stf;
            sources_cache_.push_back(std::move(def));
        }
        sources_parsed_ = true;
        return;
    }

    // Check if sources come from external file. `ext_root` must outlive
    // `sources_node` — see the note in ParseReceivers() for the lifetime
    // rationale (yaml-cpp child nodes keep a shared reference to the
    // document, and dropping that document mid-iteration manifests as
    // mysterious waveform drift on GPU builds).
    std::string source_file = GetSourceFile();
    YAML::Node ext_root;
    YAML::Node sources_node;

    if (!source_file.empty()) {
        try {
            ext_root = YAML::LoadFile(source_file);
        } catch (const YAML::Exception& e) {
            std::cerr << "Error loading source file: " << e.what() << std::endl;
            sources_parsed_ = true;
            return;
        }
        sources_node = ext_root["sources"];
    } else {
        sources_node = root_["sources"]["list"];
    }

    if (!sources_node || !sources_node.IsSequence()) {
        sources_parsed_ = true;
        return;
    }

    int dim = GetDimension();

    for (size_t i = 0; i < sources_node.size(); i++) {
        YAML::Node src = sources_node[i];
        SourceDef def;

        // id and name are optional
        def.id = src["id"] ? src["id"].as<int>() : static_cast<int>(i + 1);
        def.name = src["name"] ? src["name"].as<std::string>() : "source_" + std::to_string(def.id);
        // type is required - validated in Validate()
        def.type = src["type"].as<std::string>();

        // Location required - validated in Validate()
        YAML::Node loc = src["location"];
        for (int d = 0; d < 3; d++) {
            def.location[d] = (d < static_cast<int>(loc.size())) ? loc[d].as<real_t>() : 0.0;
        }

        // Direction (for force type) - optional, default vertical
        if (src["direction"]) {
            YAML::Node dir = src["direction"];
            for (int d = 0; d < 3; d++) {
                def.direction[d] = (d < static_cast<int>(dir.size())) ? dir[d].as<real_t>() : 0.0;
            }
        } else {
            def.direction[0] = 0.0;
            def.direction[1] = 0.0;
            def.direction[2] = 0.0;
            def.direction[dim - 1] = 1.0;  // Default: vertical
        }

        // Wavelet parameters - required - validated in Validate()
        YAML::Node wv = src["wavelet"];
        def.wavelet_type = wv["type"].as<std::string>();

        if (def.wavelet_type == "external") {
            // External STF file: only `file` is required. `frequency` and
            // `amplitude` are determined by the file content itself; if the
            // user supplies them they're used as a PPW-check hint, otherwise
            // the PPW check is skipped for this source.
            MFEM_VERIFY(wv["file"],
                "Source '" << def.name << "': wavelet.type='external' requires 'file' parameter");
            def.external_file = wv["file"].as<std::string>();
            def.frequency = wv["frequency"] ? wv["frequency"].as<real_t>() : 0.0;
            def.amplitude = 1.0;  // Not used for external
        } else {
            // Ricker or Gaussian - require frequency (PPW check) and amplitude
            def.frequency = wv["frequency"].as<real_t>();
            def.amplitude = wv["amplitude"].as<real_t>();
        }
        // delay is optional (0.0 is valid default)
        def.delay = wv["delay"] ? wv["delay"].as<real_t>() : 0.0;

        // Moment tensor (if applicable) - 0.0 is valid default
        if (def.type == "moment_tensor" && src["moment_tensor"]) {
            YAML::Node mt = src["moment_tensor"];
            def.M[0] = mt["Mxx"] ? mt["Mxx"].as<real_t>() : 0.0;
            def.M[1] = mt["Myy"] ? mt["Myy"].as<real_t>() : 0.0;
            def.M[2] = mt["Mzz"] ? mt["Mzz"].as<real_t>() : 0.0;
            def.M[3] = mt["Mxy"] ? mt["Mxy"].as<real_t>() : 0.0;
            def.M[4] = mt["Mxz"] ? mt["Mxz"].as<real_t>() : 0.0;
            def.M[5] = mt["Myz"] ? mt["Myz"].as<real_t>() : 0.0;
        }

        // Observed data (inversion mode) - optional, validated in Validate()
        if (src["observed"]) {
            def.has_observed = true;
            YAML::Node obs = src["observed"];
            def.observed.file = obs["file"] ?
                obs["file"].as<std::string>() : "";
            if (obs["resample"]) {
                YAML::Node rs = obs["resample"];
                def.observed.resample.enabled = rs["enabled"] ?
                    rs["enabled"].as<bool>() : false;
                def.observed.resample.method = rs["method"] ?
                    rs["method"].as<std::string>() : "lanczos";
                def.observed.resample.lanczos_a = rs["lanczos_a"] ?
                    rs["lanczos_a"].as<int>() : 8;
            }
        }

        sources_cache_.push_back(def);
    }

    sources_parsed_ = true;
}

std::vector<SourceDef> YamlConfig::GetAllSources() const {
    ParseSources();
    return sources_cache_;
}

// =============================================================================
// Receivers Section
// =============================================================================

bool YamlConfig::HasReceivers() const {
    return root_["receivers"].IsDefined();
}

std::string YamlConfig::GetReceiverFile() const {
    // Optional - returns empty string if not specified (sentinel value OK)
    YAML::Node rec = root_["receivers"];
    if (rec && rec["file"]) {
        return rec["file"].as<std::string>();
    }
    return "";
}

std::string YamlConfig::GetReceiverFormat() const {
    YAML::Node rec = root_["receivers"];
    if (rec && rec["format"]) {
        return rec["format"].as<std::string>();
    }
    return "yaml";
}

int YamlConfig::GetReceiverShotId() const {
    YAML::Node rec = root_["receivers"];
    if (rec && rec["shot_id"]) {
        return rec["shot_id"].as<int>();
    }
    return 0;
}

void YamlConfig::ParseReceivers() const {
    if (receivers_parsed_) return;

    receivers_cache_.clear();

    // Read parent-level type(s) (required) - validated in Validate()
    // Can be single string or list: "velocity" or ["velocity", "displacement"]
    std::vector<std::string> default_types;
    YAML::Node type_node = root_["receivers"]["type"];
    if (type_node.IsSequence()) {
        for (size_t i = 0; i < type_node.size(); i++) {
            default_types.push_back(type_node[i].as<std::string>());
        }
    } else {
        default_types.push_back(type_node.as<std::string>());
    }
    int dim = GetDimension();

    // HDF5 input branch: delegate to HDF5SourceReceiverReader and translate
    // its ReceiverConfig::Config back into ReceiverDef (the cached form).
    if (GetReceiverFormat() == "hdf5") {
        const std::string h5_path = GetReceiverFile();
        const int shot_id = GetReceiverShotId();
        ReceiverConfig::Config rc =
            HDF5SourceReceiverReader::ReadReceivers(h5_path, shot_id,
                                                    default_types, dim);
        for (const auto& r : rc.receivers) {
            ReceiverDef def;
            def.name = r.name;
            for (int d = 0; d < 3; ++d) {
                def.location[d] = (d < static_cast<int>(r.location.size()))
                                      ? r.location[d] : 0.0;
            }
            def.types = r.types;
            def.weight = r.weight;
            receivers_cache_.push_back(std::move(def));
        }
        receivers_parsed_ = true;
        return;
    }

    // 1. Determine receiver data source (external file or inline).
    // `ext_root` must outlive `rec_list_node` / `rec_line_node`: yaml-cpp
    // Node objects hold a shared reference to their document's internal
    // tree, and previous versions of this function declared `ext` inside
    // an `if` block — once `ext` went out of scope the document's
    // refcount could drop to zero, invalidating the child nodes below.
    // The user-visible symptom was a ~7% waveform divergence between
    // inline and external receiver paths on GPU builds (where the stale
    // memory happened to be reused differently than on CPU). Keeping
    // `ext_root` alive at function scope fixes the lifetime issue.
    std::string receiver_file = GetReceiverFile();
    YAML::Node ext_root;
    YAML::Node rec_list_node;
    YAML::Node rec_line_node;

    if (!receiver_file.empty()) {
        try {
            ext_root = YAML::LoadFile(receiver_file);
        } catch (const YAML::Exception& e) {
            std::cerr << "Error loading receiver file: " << e.what() << std::endl;
            receivers_parsed_ = true;
            return;
        }
        rec_list_node = ext_root["list"];
        rec_line_node = ext_root["line"];
    } else {
        rec_list_node = root_["receivers"]["list"];
        rec_line_node = root_["receivers"]["line"];
    }

    // 2. Parse receivers list
    if (rec_list_node && rec_list_node.IsSequence()) {
        for (size_t i = 0; i < rec_list_node.size(); i++) {
            YAML::Node rec = rec_list_node[i];
            ReceiverDef def;

            // name is optional
            def.name = rec["name"] ? rec["name"].as<std::string>() : "R" + std::to_string(i);
            // location required - validated in Validate() for inline, checked here for external
            YAML::Node loc = rec["location"];
            if (!loc) {
                std::cerr << "Receiver list[" << i << "] missing location" << std::endl;
                continue;
            }
            for (int d = 0; d < 3; d++) {
                def.location[d] = (d < static_cast<int>(loc.size())) ? loc[d].as<real_t>() : 0.0;
            }
            // Per-receiver `type:` (optional) overrides the parent-level
            // `receivers.type` default. Useful for coupled fluid-solid
            // simulations where a fluid-side station wants only PS and
            // a solid-side station wants DISP/VEL/ACC — the global
            // list would otherwise force both sides to try every type
            // and trip a not-found error in one submesh's locator.
            // Accepts either a scalar string or a sequence, mirroring
            // the parent-level syntax.
            YAML::Node rec_type = rec["type"];
            if (rec_type) {
                def.types.clear();
                if (rec_type.IsSequence()) {
                    for (size_t j = 0; j < rec_type.size(); ++j) {
                        def.types.push_back(rec_type[j].as<std::string>());
                    }
                } else {
                    def.types.push_back(rec_type.as<std::string>());
                }
            } else {
                def.types = default_types;
            }
            // weight is optional (1.0 is valid default for weight)
            def.weight = rec["weight"] ? rec["weight"].as<real_t>() : 1.0;

            receivers_cache_.push_back(def);
        }
    }

    // 3. Line expansion
    if (rec_line_node) {
        ReceiverLineDef grid;
        YAML::Node start_node = rec_line_node["start"];
        YAML::Node end_node = rec_line_node["end"];
        if (start_node && end_node && rec_line_node["count"]) {
            for (int d = 0; d < 3; d++) {
                grid.start[d] = (d < static_cast<int>(start_node.size())) ? start_node[d].as<real_t>() : 0.0;
                grid.end_[d] = (d < static_cast<int>(end_node.size())) ? end_node[d].as<real_t>() : 0.0;
            }
            grid.count = rec_line_node["count"].as<int>();
            grid.prefix = rec_line_node["prefix"] ? rec_line_node["prefix"].as<std::string>() : "REC";

            for (int i = 0; i < grid.count; i++) {
                ReceiverDef def;
                char buf[32];
                snprintf(buf, sizeof(buf), "%s%03d", grid.prefix.c_str(), i + 1);
                def.name = buf;
                def.types = default_types;
                def.weight = 1.0;

                real_t t = (grid.count > 1) ?
                           static_cast<real_t>(i) / (grid.count - 1) : 0.0;
                for (int d = 0; d < dim; d++) {
                    def.location[d] = grid.start[d] + t * (grid.end_[d] - grid.start[d]);
                }
                receivers_cache_.push_back(def);
            }
        }
    }

    receivers_parsed_ = true;
}

std::vector<ReceiverDef> YamlConfig::GetAllReceivers() const {
    ParseReceivers();
    return receivers_cache_;
}

bool YamlConfig::HasReceiverLine() const {
    return root_["receivers"]["line"].IsDefined();
}

ReceiverLineDef YamlConfig::GetReceiverLine() const {
    // Required when line defined - validated in Validate()
    ReceiverLineDef grid;
    YAML::Node g = root_["receivers"]["line"];

    // All grid parameters required - validated in Validate()
    YAML::Node start_node = g["start"];
    YAML::Node end_node = g["end"];
    for (int d = 0; d < 3; d++) {
        grid.start[d] = (d < static_cast<int>(start_node.size())) ? start_node[d].as<real_t>() : 0.0;
        grid.end_[d] = (d < static_cast<int>(end_node.size())) ? end_node[d].as<real_t>() : 0.0;
    }
    grid.count = g["count"].as<int>();
    grid.prefix = g["prefix"].as<std::string>();

    return grid;
}

std::vector<std::string> YamlConfig::GetReceiverOutputFormats() const {
    // Canonical form, validated in Validate():
    //   `receivers.output.formats: [{type: ascii}, {type: su}, ...]`
    std::vector<std::string> formats;
    YAML::Node fmt_node = root_["receivers"]["output"]["formats"];
    for (const auto& item : fmt_node) {
        formats.push_back(item["type"].as<std::string>());
    }
    return formats;
}

std::string YamlConfig::GetReceiverOutputFilename() const {
    if (root_["receivers"]["output"]["filename"]) {
        return root_["receivers"]["output"]["filename"].as<std::string>();
    }
    return "";
}

// =============================================================================
// Device Section (replaces Parallel Section)
// =============================================================================

std::string YamlConfig::GetDevice() const {
    // Required - validated in Validate()
    return root_["device"]["type"].as<std::string>();
}

int YamlConfig::GetSeismoBufferSteps() const {
    // Only available in new "device" section, default 0 (all steps)
    if (root_["device"].IsDefined() && root_["device"]["seismo_buffer_steps"]) {
        return root_["device"]["seismo_buffer_steps"].as<int>();
    }
    return 0;  // Default: buffer all steps
}

// =============================================================================
// Simulation Mode
// =============================================================================

std::string YamlConfig::GetSimulationMode() const {
    YAML::Node sim = root_["simulation"];
    if (sim && sim["mode"]) {
        return sim["mode"].as<std::string>();
    }
    return "forward";
}

// =============================================================================
// Inversion Parameters (simulation direct)
// =============================================================================

std::string YamlConfig::GetMisfitType() const {
    YAML::Node sim = root_["simulation"];
    if (sim && sim["misfit_type"]) {
        return sim["misfit_type"].as<std::string>();
    }
    return "l2_waveform";
}

int YamlConfig::GetNumCheckpoints() const {
    YAML::Node sim = root_["simulation"];
    if (sim && sim["num_checkpoints"]) {
        return sim["num_checkpoints"].as<int>();
    }
    return 10;  // default
}

std::string YamlConfig::GetCheckpointStorage() const {
    YAML::Node sim = root_["simulation"];
    if (sim && sim["checkpoint_storage"]) {
        return sim["checkpoint_storage"].as<std::string>();
    }
    return "memory";
}

std::string YamlConfig::GetCheckpointDevice() const {
    YAML::Node sim = root_["simulation"];
    if (sim && sim["checkpoint_device"]) {
        return sim["checkpoint_device"].as<std::string>();
    }
    return "auto";
}

std::string YamlConfig::GetCheckpointDir() const {
    YAML::Node sim = root_["simulation"];
    if (sim && sim["checkpoint_dir"]) {
        return sim["checkpoint_dir"].as<std::string>();
    }
    return "./checkpoints/";
}

std::string YamlConfig::GetKernelOutputDir() const {
    YAML::Node sim = root_["simulation"];
    if (sim && sim["kernel_output_dir"]) {
        return sim["kernel_output_dir"].as<std::string>();
    }
    return "./kernels/";
}

std::string YamlConfig::GetSensitivityBackend() const {
    YAML::Node inv = root_["inversion"];
    if (inv && inv["sensitivity"] && inv["sensitivity"]["backend"]) {
        std::string b = inv["sensitivity"]["backend"].as<std::string>();
        if (b != "hand" && b != "ad") {
            MFEM_ABORT("Invalid inversion.sensitivity.backend: '" << b
                       << "' (valid: hand, ad)");
        }
        return b;
    }
    return "hand";  // default to production hand path
}

bool YamlConfig::GetInvertQ() const {
    YAML::Node inv = root_["inversion"];
    if (inv && inv["invert_Q"]) {
        return inv["invert_Q"].as<bool>();
    }
    return false;
}

}  // namespace SEM
