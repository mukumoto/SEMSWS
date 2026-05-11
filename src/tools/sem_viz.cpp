/**
 * @file sem_viz.cpp
 * @brief Offline visualisation tool: ADIOS2 .bp (MaterialField layout) → GLVis / ParaView / GMT
 *
 * Reads one or more BP files that were written in the SEM `MaterialField`
 * layout (materials and kernels both use this layout), reconstructs them
 * as `ParGridFunction`s on the mesh/FES described by the YAML config, and
 * emits visualisation files using the same writer backend that the
 * runtime `MaterialWriter` uses. This guarantees that offline kernel /
 * material plots match exactly what the simulator would produce at
 * runtime — no ad-hoc coordinate reconstruction in a Python script.
 *
 * Typical usage (2D):
 *   mpirun -np N build/src/sem_viz \
 *       --config config_inversion_hand.yaml \
 *       --input kernel_vp=output_hand/kernels/kernel_vp_src001.bp \
 *       --input kernel_rho=output_hand/kernels/kernel_rho_src001.bp \
 *       --output-dir viz_hand \
 *       --format paraview,glvis,gmt \
 *       --resolution 400,400
 *
 * Typical usage (3D with cross-sections):
 *   mpirun -np N build/src/sem_viz \
 *       --config run3d.yaml \
 *       --input vp=kernel_vp_src001.bp \
 *       --output-dir viz \
 *       --format gmt \
 *       --resolution 400,200 \
 *       --cross-sections 'xz=5000;xy=-1000'
 *
 * MPI: must be invoked with the same `-np N` used when writing the .bp.
 */

#include "config/ConfigLoaders.hpp"
#include "config/YamlConfig.hpp"
#include "io/ADIOS2IO.hpp"
#include "io/FieldVizWriter.hpp"
#include "material/MaterialField.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace SEM;
using namespace mfem;

namespace {

struct NamedInput {
    std::string name;   // display name used in viz outputs
    std::string path;   // .bp path
};

struct CliArgs {
    std::string config_file;
    std::string output_dir = "viz";
    std::string device_str = "cpu";
    std::string var_name = "data";               // BP variable name
    std::vector<std::string> formats;            // subset of {"paraview","glvis","gmt"}
    std::array<int, 2> resolution = {200, 200};
    int refinement = 1;
    std::string data_format = "binary32";
    int compression = -1;
    std::vector<NamedInput> inputs;
    GMTCrossSections cross_sections;
};

std::vector<std::string> SplitCsv(const std::string& s, char sep = ',') {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, sep)) {
        if (!tok.empty()) out.push_back(tok);
    }
    return out;
}

// Parse "name=path". If no "=" is present, derive `name` from the file
// stem (strip directory and .bp suffix).
NamedInput ParseInputSpec(const std::string& spec) {
    auto eq = spec.find('=');
    if (eq != std::string::npos) {
        return {spec.substr(0, eq), spec.substr(eq + 1)};
    }
    std::string name = spec;
    auto slash = name.find_last_of('/');
    if (slash != std::string::npos) name = name.substr(slash + 1);
    auto dot = name.rfind(".bp");
    if (dot != std::string::npos) name = name.substr(0, dot);
    return {name, spec};
}

// Parse "yz=1,2;xz=3;xy=-1000" into a GMTCrossSections.
GMTCrossSections ParseCrossSections(const std::string& spec) {
    GMTCrossSections cs;
    for (const auto& group : SplitCsv(spec, ';')) {
        auto eq = group.find('=');
        if (eq == std::string::npos) continue;
        std::string plane = group.substr(0, eq);
        std::vector<real_t>* target = nullptr;
        if (plane == "yz") target = &cs.yz;
        else if (plane == "xz") target = &cs.xz;
        else if (plane == "xy") target = &cs.xy;
        else continue;
        for (const auto& v : SplitCsv(group.substr(eq + 1), ',')) {
            target->push_back(std::stod(v));
        }
    }
    return cs;
}

void PrintUsage() {
    std::cerr <<
"Usage: mpirun -np N sem_viz --config C.yaml [options] --input name=X.bp [...]\n\n"
"Required:\n"
"  --config <yaml>            Config used when generating the .bp files\n"
"  --input  <name=path.bp>    Repeatable. 'name' becomes the field label\n"
"                             in viz outputs. If '=' is omitted, the name\n"
"                             is derived from the filename stem.\n\n"
"Common options:\n"
"  --output-dir <path>        Output root (default: viz)\n"
"  --format <list>            Comma-separated subset of\n"
"                             {paraview,glvis,gmt,all} (default: all)\n"
"  --resolution <nx,ny>       GMT grid size (default: 200,200)\n"
"  --refinement <k>           ParaView element subdivision (default: 1)\n"
"  --data-format <fmt>        ParaView data encoding:\n"
"                             ascii|binary|binary32 (default: binary32)\n"
"  --compression <n>          ParaView zlib level 0-9, -1=default\n"
"  --cross-sections <spec>    3D GMT slice spec, e.g.\n"
"                             'yz=5000;xz=5000;xy=-1000'\n"
"  --var <name>               BP variable name (default: data)\n"
"  --device <cpu|cuda>        (default: cpu)\n";
}

bool ParseCli(int argc, char** argv, CliArgs& out) {
    std::string fmt_spec = "all";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                if (Mpi::Root()) std::cerr << "missing value for " << a << "\n";
                return {};
            }
            return argv[++i];
        };
        if (a == "--config")            out.config_file = next();
        else if (a == "--input")        out.inputs.push_back(ParseInputSpec(next()));
        else if (a == "--output-dir")   out.output_dir = next();
        else if (a == "--format")       fmt_spec = next();
        else if (a == "--resolution") {
            auto parts = SplitCsv(next());
            if (parts.size() == 2) {
                out.resolution[0] = std::stoi(parts[0]);
                out.resolution[1] = std::stoi(parts[1]);
            }
        }
        else if (a == "--refinement")   out.refinement = std::stoi(next());
        else if (a == "--data-format")  out.data_format = next();
        else if (a == "--compression")  out.compression = std::stoi(next());
        else if (a == "--cross-sections") out.cross_sections = ParseCrossSections(next());
        else if (a == "--var")          out.var_name = next();
        else if (a == "--device")       out.device_str = next();
        else if (a == "-h" || a == "--help") { PrintUsage(); return false; }
        else {
            if (Mpi::Root()) std::cerr << "unknown argument: " << a << "\n";
            return false;
        }
    }

    if (out.config_file.empty() || out.inputs.empty()) {
        if (Mpi::Root()) PrintUsage();
        return false;
    }

    // Expand format list
    if (fmt_spec == "all") {
        out.formats = {"paraview", "glvis", "gmt"};
    } else {
        out.formats = SplitCsv(fmt_spec);
    }
    return true;
}

}  // anonymous namespace

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    Hypre::Init();

    CliArgs args;
    if (!ParseCli(argc, argv, args)) {
        return 1;
    }

    Device device(args.device_str);
    MPI_Comm comm = MPI_COMM_WORLD;

    // ---- Load config / mesh / FES -----------------------------------------
    YamlConfig config(args.config_file);
    if (!config.IsValid()) {
        if (Mpi::Root()) {
            std::cerr << "Config error: " << config.GetValidationError() << std::endl;
        }
        return 1;
    }

    const int order = config.GetOrder();
    const int dim   = config.GetDimension();

    std::unique_ptr<ParMesh> pmesh(LoadParMesh(config, comm));
    pmesh->SetCurvature(order, true, dim, Ordering::byNODES);

    H1_FECollection fec(order, dim, BasisType::GaussLobatto);
    ParFiniteElementSpace fes(pmesh.get(), &fec);

    // ---- Load each BP input as a ParGridFunction --------------------------
    // Kept in parallel vectors so the writer API's expected layout
    // (vector<pair<string, ParGridFunction*>>) can be built cheaply.
    std::vector<std::unique_ptr<ParGridFunction>> owned;
    FieldVizWriter::FieldList fields;
    owned.reserve(args.inputs.size());

    for (const auto& in : args.inputs) {
        if (Mpi::Root()) {
            std::cout << "Loading [" << in.name << "] from " << in.path << "\n";
        }
        auto gf = std::make_unique<ParGridFunction>(&fes);
        if (dim == 2) {
            MaterialField f = LoadFieldBP(in.path, args.var_name, comm);
            f.ToParGridFunction(fes, *gf);
        } else {
            MaterialField3D f = LoadFieldBP3D(in.path, args.var_name, comm);
            f.ToParGridFunction(fes, *gf);
        }
        fields.emplace_back(in.name, gf.get());
        owned.push_back(std::move(gf));
    }

    // ---- Build a single OutputFormatConfig reused across formats ----------
    OutputFormatConfig fmt;
    fmt.refinement      = args.refinement;
    fmt.data_format     = args.data_format;
    fmt.compression     = args.compression;
    fmt.resolution      = args.resolution;
    fmt.cross_sections  = args.cross_sections;

    // ---- Dispatch ----------------------------------------------------------
    for (const auto& f : args.formats) {
        fmt.type = f;
        if (f == "paraview") {
            FieldVizWriter::WriteParaView(args.output_dir, "", fields, fmt, pmesh.get());
        } else if (f == "glvis") {
            FieldVizWriter::WriteGLVis(args.output_dir, "", fields);
        } else if (f == "gmt") {
            FieldVizWriter::WriteGMT(args.output_dir, "", fields, fmt, *pmesh, comm);
        } else if (Mpi::Root()) {
            std::cerr << "sem_viz: unknown format '" << f << "' (skipped)\n";
        }
    }

    if (Mpi::Root()) {
        std::cout << "sem_viz: wrote to " << args.output_dir << " (formats:";
        for (const auto& f : args.formats) std::cout << " " << f;
        std::cout << ")" << std::endl;
    }

    return 0;
}
