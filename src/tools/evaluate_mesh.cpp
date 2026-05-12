/**
 * @file evaluate_mesh.cpp
 * @brief MPI-parallel mesh quality evaluation tool
 *
 * Reads a YAML config (same format as test_forward_simulation), loads mesh
 * and material via FromConfig() pipeline, and evaluates per-element mesh
 * quality metrics: element size, CFL dt_max, and PPW.
 *
 * Outputs summary statistics and optional histogram CSV files.
 *
 * Usage:
 *   mpirun -np N ./evaluate_mesh --config config.yaml [options]
 */

#include <mfem.hpp>
#include "config/YamlConfig.hpp"
#include "config/ConfigLoaders.hpp"
#include "config/ConfigTypes.hpp"
#include "common/Types.hpp"
#include "material/isotropic_elastic/IsotropicElasticMaterial.hpp"
#include "material/isotropic_acoustic/IsotropicAcousticMaterial.hpp"
#include "material/MaterialBase.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <map>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace mfem;
using namespace SEM;

// =============================================================================
// ComputeElementMinGLLDistance (copied from SimulationFacade.cpp)
// =============================================================================

template<int Dim>
real_t ComputeElementMinGLLDistance(ParMesh& mesh, int e, int order) {
    const int ngll = order + 1;
    ElementTransformation* T = mesh.GetElementTransformation(e);
    static IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);
    const IntegrationRule& ir = gll_rules.Get(
        mesh.GetElementGeometry(e), 2 * order - 1);
    DenseMatrix pts(Dim, ir.GetNPoints());
    T->Transform(ir, pts);

    real_t min_dist_sq = std::numeric_limits<real_t>::max();

    if constexpr (Dim == 2) {
        for (int j = 0; j < ngll; j++) {
            for (int i = 0; i < ngll; i++) {
                int idx = i + j * ngll;
                if (i < ngll - 1) {
                    int idx_x = (i + 1) + j * ngll;
                    real_t dist_sq = 0.0;
                    for (int d = 0; d < Dim; d++) {
                        real_t diff = pts(d, idx_x) - pts(d, idx);
                        dist_sq += diff * diff;
                    }
                    min_dist_sq = std::min(min_dist_sq, dist_sq);
                }
                if (j < ngll - 1) {
                    int idx_y = i + (j + 1) * ngll;
                    real_t dist_sq = 0.0;
                    for (int d = 0; d < Dim; d++) {
                        real_t diff = pts(d, idx_y) - pts(d, idx);
                        dist_sq += diff * diff;
                    }
                    min_dist_sq = std::min(min_dist_sq, dist_sq);
                }
            }
        }
    } else {
        for (int k = 0; k < ngll; k++) {
            for (int j = 0; j < ngll; j++) {
                for (int i = 0; i < ngll; i++) {
                    int idx = i + j * ngll + k * ngll * ngll;
                    if (i < ngll - 1) {
                        int idx_x = (i + 1) + j * ngll + k * ngll * ngll;
                        real_t dist_sq = 0.0;
                        for (int d = 0; d < Dim; d++) {
                            real_t diff = pts(d, idx_x) - pts(d, idx);
                            dist_sq += diff * diff;
                        }
                        min_dist_sq = std::min(min_dist_sq, dist_sq);
                    }
                    if (j < ngll - 1) {
                        int idx_y = i + (j + 1) * ngll + k * ngll * ngll;
                        real_t dist_sq = 0.0;
                        for (int d = 0; d < Dim; d++) {
                            real_t diff = pts(d, idx_y) - pts(d, idx);
                            dist_sq += diff * diff;
                        }
                        min_dist_sq = std::min(min_dist_sq, dist_sq);
                    }
                    if (k < ngll - 1) {
                        int idx_z = i + j * ngll + (k + 1) * ngll * ngll;
                        real_t dist_sq = 0.0;
                        for (int d = 0; d < Dim; d++) {
                            real_t diff = pts(d, idx_z) - pts(d, idx);
                            dist_sq += diff * diff;
                        }
                        min_dist_sq = std::min(min_dist_sq, dist_sq);
                    }
                }
            }
        }
    }
    return std::sqrt(min_dist_sq);
}

// =============================================================================
// Build material (same pattern as refine_mesh)
// =============================================================================

template<int Dim>
std::unique_ptr<MaterialBase> BuildMaterial(
    const YamlConfig& config, ParMesh& mesh, int order) {

    H1_FECollection fec(order, Dim);
    ParFiniteElementSpace fes(&mesh, &fec);

    IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);
    const int exact = 2 * order - 1;
    IntegrationRule ir_1d = gll_rules.Get(Geometry::SEGMENT, exact);
    IntegrationRule ir;
    if constexpr (Dim == 2) {
        ir = IntegrationRule(ir_1d, ir_1d);
    } else {
        ir = IntegrationRule(ir_1d, ir_1d, ir_1d);
    }

    MaterialConfig mat_config = LoadMaterialConfig(config);
    MaterialType mat_type = StringToMaterialType(mat_config.material_type);

    if constexpr (Dim == 2) {
        if (mat_type == MaterialType::IsotropicElastic) {
            auto mat = IsotropicElasticMaterial::FromConfig(mat_config, fes, ir);
            return mat;
        } else {
            auto mat = IsotropicAcousticMaterial::FromConfig(mat_config, fes, ir);
            return mat;
        }
    } else {
        if (mat_type == MaterialType::IsotropicElastic) {
            auto mat = IsotropicElasticMaterial3D::FromConfig(mat_config, fes, ir);
            return mat;
        } else {
            auto mat = IsotropicAcousticMaterial3D::FromConfig(mat_config, fes, ir);
            return mat;
        }
    }
}

// =============================================================================
// Statistics helpers
// =============================================================================

struct MetricStats {
    real_t min_val = std::numeric_limits<real_t>::max();
    real_t max_val = std::numeric_limits<real_t>::lowest();
    double sum = 0.0;
    double sum_sq = 0.0;
    long long count = 0;

    void Add(real_t val) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
        sum_sq += static_cast<double>(val) * val;
        count++;
    }

    // MPI reduce into global stats
    MetricStats ReduceGlobal(MPI_Comm comm) const {
        MetricStats global;
        MPI_Allreduce(&min_val, &global.min_val, 1, MFEM_MPI_REAL_T, MPI_MIN, comm);
        MPI_Allreduce(&max_val, &global.max_val, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);
        MPI_Allreduce(&sum, &global.sum, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&sum_sq, &global.sum_sq, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(&count, &global.count, 1, MPI_LONG_LONG, MPI_SUM, comm);
        return global;
    }

    double Mean() const { return (count > 0) ? sum / count : 0.0; }
    double Std() const {
        if (count <= 1) return 0.0;
        double mean = Mean();
        return std::sqrt(sum_sq / count - mean * mean);
    }
};

// =============================================================================
// Histogram
// =============================================================================

struct Histogram {
    int nbins;
    real_t bin_min, bin_max, bin_width;
    std::vector<long long> counts;

    Histogram(int nbins_, real_t min_val, real_t max_val)
        : nbins(nbins_), bin_min(min_val), bin_max(max_val),
          counts(nbins_, 0) {
        bin_width = (max_val - min_val) / nbins;
        if (bin_width <= 0) bin_width = 1.0;  // safety
    }

    void Add(real_t val) {
        int bin = static_cast<int>((val - bin_min) / bin_width);
        bin = std::max(0, std::min(nbins - 1, bin));
        counts[bin]++;
    }

    Histogram ReduceGlobal(MPI_Comm comm) const {
        Histogram global(nbins, bin_min, bin_max);
        MPI_Allreduce(counts.data(), global.counts.data(), nbins, MPI_LONG_LONG, MPI_SUM, comm);
        return global;
    }

    void WriteCSV(const std::string& filename, const std::string& metric_name,
                  const std::string& config_file, real_t max_freq,
                  long long total_elements) const {
        std::ofstream ofs(filename);
        ofs << "# evaluate_mesh histogram: " << metric_name << "\n";
        ofs << "# config: " << config_file << ", f_max: " << max_freq << " Hz\n";
        ofs << "# bins: " << nbins << ", total_elements: " << total_elements << "\n";
        ofs << "bin_low,bin_high,count,fraction\n";
        for (int i = 0; i < nbins; i++) {
            real_t low = bin_min + i * bin_width;
            real_t high = low + bin_width;
            double frac = (total_elements > 0) ?
                static_cast<double>(counts[i]) / total_elements : 0.0;
            ofs << low << "," << high << "," << counts[i] << "," << frac << "\n";
        }
        ofs.close();
    }
};

// =============================================================================
// Per-attribute stats
// =============================================================================

struct AttrStats {
    long long count = 0;
    real_t ppw_min = std::numeric_limits<real_t>::max();
    real_t dt_min = std::numeric_limits<real_t>::max();
    real_t v_max = 0.0;
    real_t v_min = std::numeric_limits<real_t>::max();
};

// =============================================================================
// Main evaluation function
// =============================================================================

template<int Dim>
void EvaluateMesh(ParMesh& mesh, MaterialBase& material,
                   int order, real_t max_freq, real_t ppw_target,
                   real_t cfl_factor, MPI_Comm comm,
                   const std::string& histogram_prefix, int nbins,
                   const std::string& config_file) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int ne = mesh.GetNE();
    int ngll = order + 1;

    // --- Collect per-element metrics ---
    IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);
    MetricStats h_stats, dt_stats, ppw_stats, aspr_stats, detJ_stats;
    std::vector<real_t> h_vals(ne), dt_vals(ne), ppw_vals(ne);
    std::vector<real_t> aspr_vals(ne), detJ_vals(ne);
    std::map<int, AttrStats> attr_stats;

    // Track worst PPW element
    real_t worst_ppw_local = std::numeric_limits<real_t>::max();
    int worst_ppw_elem = -1;

    // Jacobian-based quality metrics (MFEM mesh-quality miniapp pattern)
    DenseMatrix jacobian(Dim);

    for (int e = 0; e < ne; e++) {
        real_t h = mesh.GetElementSize(e, 2);
        real_t gll_dist = ComputeElementMinGLLDistance<Dim>(mesh, e, order);
        real_t v_max = material.GetElementMaxVelocity(e);
        real_t v_min = material.GetElementMinVelocity(e);

        real_t dt_cfl = cfl_factor * gll_dist / v_max;
        real_t ppw_elem = ngll * v_min / (max_freq * h);

        // Compute Jacobian-based quality metrics at ALL DOFs (GLL nodes).
        // Report worst (max aspect ratio, min det).
        Geometry::Type geom = mesh.GetElementBaseGeometry(e);
        const IntegrationRule &ir = gll_rules.Get(geom, 2 * order - 1);
        int npts = ir.GetNPoints();

        real_t aspr_elem = 1.0;
        real_t detJ_elem_min = std::numeric_limits<real_t>::max();

        for (int q = 0; q < npts; q++) {
            const IntegrationPoint &ip = ir.IntPoint(q);
            mesh.GetElementJacobian(e, jacobian, &ip);

            // Aspect ratio: sv_max / sv_min
            real_t sv_max = jacobian.CalcSingularvalue(0);
            real_t sv_min = jacobian.CalcSingularvalue(Dim - 1);
            real_t aspr_q = (sv_min > 0) ? sv_max / sv_min : 1e30;
            aspr_elem = std::max(aspr_elem, aspr_q);

            real_t det_q = jacobian.Det();
            detJ_elem_min = std::min(detJ_elem_min, det_q);
        }

        real_t detJ_elem = detJ_elem_min;

        h_vals[e] = h;
        dt_vals[e] = dt_cfl;
        ppw_vals[e] = ppw_elem;
        aspr_vals[e] = aspr_elem;
        detJ_vals[e] = detJ_elem;

        h_stats.Add(h);
        dt_stats.Add(dt_cfl);
        ppw_stats.Add(ppw_elem);
        aspr_stats.Add(aspr_elem);
        detJ_stats.Add(detJ_elem);

        // Track worst PPW
        if (ppw_elem < worst_ppw_local) {
            worst_ppw_local = ppw_elem;
            worst_ppw_elem = e;
        }

        // Per-attribute stats
        int attr = mesh.GetAttribute(e);
        auto& as = attr_stats[attr];
        as.count++;
        as.ppw_min = std::min(as.ppw_min, ppw_elem);
        as.dt_min = std::min(as.dt_min, dt_cfl);
        as.v_max = std::max(as.v_max, v_max);
        as.v_min = std::min(as.v_min, v_min);
    }

    // --- Global reduction ---
    MetricStats h_global = h_stats.ReduceGlobal(comm);
    MetricStats dt_global = dt_stats.ReduceGlobal(comm);
    MetricStats ppw_global = ppw_stats.ReduceGlobal(comm);
    MetricStats aspr_global = aspr_stats.ReduceGlobal(comm);
    MetricStats detJ_global = detJ_stats.ReduceGlobal(comm);

    // Count PPW violations
    int local_ppw_violations = 0;
    for (int e = 0; e < ne; e++) {
        if (ppw_vals[e] < ppw_target) local_ppw_violations++;
    }
    int global_ppw_violations = 0;
    MPI_Allreduce(&local_ppw_violations, &global_ppw_violations, 1, MPI_INT, MPI_SUM, comm);

    // Find global worst PPW element
    struct { double ppw; int rank; } local_worst, global_worst;
    local_worst.ppw = worst_ppw_local;
    local_worst.rank = rank;
    MPI_Allreduce(&local_worst, &global_worst, 1, MPI_DOUBLE_INT, MPI_MINLOC, comm);

    // --- Print summary (rank 0) ---
    if (rank == 0) {
        std::cout << "\nMesh Evaluation Report\n";
        std::cout << "======================\n";
        std::cout << "Mesh: " << h_global.count << " elements (global), dimension " << Dim << "\n";
        std::cout << "Order: " << order << ", CFL factor: " << cfl_factor
                  << ", f_max: " << max_freq << " Hz, PPW target: " << ppw_target << "\n";

        std::cout << "\nPer-element statistics:\n";
        std::cout << std::setw(18) << "" << std::setw(12) << "min"
                  << std::setw(12) << "max" << std::setw(12) << "mean"
                  << std::setw(12) << "std" << "\n";

        auto print_row = [](const char* name, const MetricStats& s) {
            std::cout << "  " << std::setw(14) << std::left << name << std::right
                      << std::setw(12) << std::scientific << std::setprecision(3) << s.min_val
                      << std::setw(12) << s.max_val
                      << std::setw(12) << s.Mean()
                      << std::setw(12) << s.Std() << "\n";
        };

        print_row("h [m]", h_global);
        print_row("dt_CFL [s]", dt_global);
        print_row("PPW", ppw_global);
        print_row("Aspect ratio", aspr_global);
        print_row("det(J)", detJ_global);

        std::cout << "\nCFL summary:\n";
        std::cout << "  dt_max (global): " << std::scientific << std::setprecision(4)
                  << dt_global.min_val << " s\n";

        std::cout << "\nPPW summary:\n";
    }

    // Print worst PPW element details (from the rank that owns it)
    if (rank == global_worst.rank && worst_ppw_elem >= 0) {
        Vector centroid(Dim);
        mesh.GetElementCenter(worst_ppw_elem, centroid);
        std::cout << "  Worst PPW: " << std::fixed << std::setprecision(2)
                  << worst_ppw_local << " at centroid (";
        for (int d = 0; d < Dim; d++) {
            if (d > 0) std::cout << ", ";
            std::cout << centroid(d);
        }
        std::cout << ")\n";
    }
    MPI_Barrier(comm);

    if (rank == 0) {
        std::cout << "  PPW < " << std::fixed << std::setprecision(1) << ppw_target
                  << ": " << global_ppw_violations << " / " << h_global.count
                  << " (" << std::fixed << std::setprecision(2)
                  << (100.0 * global_ppw_violations / h_global.count) << "%)\n";
    }

    // --- Per-attribute breakdown ---
    // Gather all unique attributes
    std::vector<int> local_attrs;
    for (const auto& [attr, _] : attr_stats) {
        local_attrs.push_back(attr);
    }
    // Collect from all ranks
    int local_nattr = local_attrs.size();
    int max_nattr;
    MPI_Allreduce(&local_nattr, &max_nattr, 1, MPI_INT, MPI_MAX, comm);

    // Simple approach: reduce per-attribute stats for known attributes
    // Gather all attributes to rank 0
    std::set<int> all_attrs_set;
    {
        // Broadcast local attrs to all ranks via Allgather
        int nattr_total;
        std::vector<int> nattr_per_rank(Mpi::WorldSize());
        MPI_Allgather(&local_nattr, 1, MPI_INT,
                       nattr_per_rank.data(), 1, MPI_INT, comm);
        nattr_total = 0;
        std::vector<int> displs(Mpi::WorldSize());
        for (int r = 0; r < Mpi::WorldSize(); r++) {
            displs[r] = nattr_total;
            nattr_total += nattr_per_rank[r];
        }
        std::vector<int> all_attrs(nattr_total);
        MPI_Allgatherv(local_attrs.data(), local_nattr, MPI_INT,
                         all_attrs.data(), nattr_per_rank.data(),
                         displs.data(), MPI_INT, comm);
        for (int a : all_attrs) all_attrs_set.insert(a);
    }

    if (rank == 0) {
        std::cout << "\nPer-attribute breakdown:\n";
    }
    for (int attr : all_attrs_set) {
        AttrStats local_as;
        if (attr_stats.count(attr)) {
            local_as = attr_stats[attr];
        }
        // Reduce
        long long global_count;
        real_t global_ppw_min_attr, global_dt_min_attr, global_v_max_attr, global_v_min_attr;
        MPI_Allreduce(&local_as.count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, comm);
        MPI_Allreduce(&local_as.ppw_min, &global_ppw_min_attr, 1, MFEM_MPI_REAL_T, MPI_MIN, comm);
        MPI_Allreduce(&local_as.dt_min, &global_dt_min_attr, 1, MFEM_MPI_REAL_T, MPI_MIN, comm);
        MPI_Allreduce(&local_as.v_max, &global_v_max_attr, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);
        MPI_Allreduce(&local_as.v_min, &global_v_min_attr, 1, MFEM_MPI_REAL_T, MPI_MIN, comm);

        if (rank == 0 && global_count > 0) {
            std::cout << "  Attr " << attr << ": " << global_count << " elems"
                      << ", Vp=" << std::fixed << std::setprecision(0) << global_v_max_attr
                      << ", Vs=" << global_v_min_attr
                      << ", PPW_min=" << std::fixed << std::setprecision(1) << global_ppw_min_attr
                      << ", dt_min=" << std::scientific << std::setprecision(2) << global_dt_min_attr
                      << "\n";
        }
    }

    // --- Histogram output ---
    if (!histogram_prefix.empty()) {
        // Build histograms using global min/max
        Histogram h_hist(nbins, h_global.min_val, h_global.max_val);
        Histogram dt_hist(nbins, dt_global.min_val, dt_global.max_val);
        Histogram ppw_hist(nbins, ppw_global.min_val, ppw_global.max_val);
        Histogram aspr_hist(nbins, aspr_global.min_val, aspr_global.max_val);
        Histogram detJ_hist(nbins, detJ_global.min_val, detJ_global.max_val);

        for (int e = 0; e < ne; e++) {
            h_hist.Add(h_vals[e]);
            dt_hist.Add(dt_vals[e]);
            ppw_hist.Add(ppw_vals[e]);
            aspr_hist.Add(aspr_vals[e]);
            detJ_hist.Add(detJ_vals[e]);
        }

        Histogram h_global_hist = h_hist.ReduceGlobal(comm);
        Histogram dt_global_hist = dt_hist.ReduceGlobal(comm);
        Histogram ppw_global_hist = ppw_hist.ReduceGlobal(comm);
        Histogram aspr_global_hist = aspr_hist.ReduceGlobal(comm);
        Histogram detJ_global_hist = detJ_hist.ReduceGlobal(comm);

        if (rank == 0) {
            h_global_hist.WriteCSV(histogram_prefix + "_element_size.csv",
                                    "element_size", config_file, max_freq, h_global.count);
            dt_global_hist.WriteCSV(histogram_prefix + "_dt_cfl.csv",
                                     "dt_CFL", config_file, max_freq, h_global.count);
            ppw_global_hist.WriteCSV(histogram_prefix + "_ppw.csv",
                                      "PPW", config_file, max_freq, h_global.count);
            aspr_global_hist.WriteCSV(histogram_prefix + "_aspect_ratio.csv",
                                       "aspect_ratio", config_file, max_freq, h_global.count);
            detJ_global_hist.WriteCSV(histogram_prefix + "_jacobian_det.csv",
                                       "jacobian_det", config_file, max_freq, h_global.count);

            std::cout << "\nHistogram CSV files written:\n";
            std::cout << "  " << histogram_prefix << "_element_size.csv\n";
            std::cout << "  " << histogram_prefix << "_dt_cfl.csv\n";
            std::cout << "  " << histogram_prefix << "_ppw.csv\n";
            std::cout << "  " << histogram_prefix << "_aspect_ratio.csv\n";
            std::cout << "  " << histogram_prefix << "_jacobian_det.csv\n";
        }
    }
}

// =============================================================================
// Usage
// =============================================================================

void PrintUsage(const char* prog) {
    std::cout << "Usage: mpirun -np N " << prog << " --config config.yaml [options]\n\n"
              << "Required:\n"
              << "  --config <file>        YAML config file (mesh, material, order, mesh.max_freq, mesh.ppw)\n\n"
              << "Optional:\n"
              << "  --cfl <factor>         CFL factor override (default: from config)\n"
              << "  --histogram <prefix>   Output histogram CSV files with this prefix\n"
              << "  --nbins <N>            Number of histogram bins (default: 50)\n"
              << "  -h, --help             Show this help\n\n"
              << "Output:\n"
              << "  Summary statistics (console)\n"
              << "  <prefix>_element_size.csv  Element size histogram\n"
              << "  <prefix>_dt_cfl.csv        CFL dt_max histogram\n"
              << "  <prefix>_ppw.csv           PPW histogram\n"
              << "  <prefix>_aspect_ratio.csv  Aspect ratio histogram\n"
              << "  <prefix>_jacobian_det.csv  Jacobian determinant histogram\n";
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    Mpi::Init(argc, argv);
    int rank = Mpi::WorldRank();

    // --- Parse arguments ---
    std::string config_file;
    std::string histogram_prefix;
    int nbins = 50;
    real_t cfl_override = -1.0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--config") && i + 1 < argc) {
            config_file = argv[++i];
        } else if ((arg == "--cfl") && i + 1 < argc) {
            cfl_override = std::atof(argv[++i]);
        } else if ((arg == "--histogram") && i + 1 < argc) {
            histogram_prefix = argv[++i];
        } else if ((arg == "--nbins") && i + 1 < argc) {
            nbins = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            if (rank == 0) PrintUsage(argv[0]);
            Mpi::Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::cerr << "Unknown option: " << arg << std::endl;
                PrintUsage(argv[0]);
            }
            Mpi::Finalize();
            return 1;
        }
    }

    if (config_file.empty()) {
        if (rank == 0) {
            std::cerr << "Error: --config is required.\n\n";
            PrintUsage(argv[0]);
        }
        Mpi::Finalize();
        return 1;
    }

    // --- Load config ---
    if (rank == 0) std::cout << "Loading config: " << config_file << std::endl;
    YamlConfig config(config_file);
    if (!config.IsValid()) {
        if (rank == 0) {
            std::cerr << "Error: Invalid config: " << config.GetValidationError() << std::endl;
        }
        Mpi::Finalize();
        return 1;
    }

    int order = config.GetOrder();
    int dim = config.GetDimension();
    real_t max_freq = config.GetMaxFreq();
    real_t ppw_target = config.GetPPW();
    real_t cfl_factor = (cfl_override > 0) ? cfl_override : config.GetCflFactor();

    if (rank == 0) {
        std::cout << "  Material: " << config.GetMaterialType() << "\n";
    }

    // --- Load mesh ---
    if (rank == 0) std::cout << "Loading mesh..." << std::endl;
    std::unique_ptr<ParMesh> pmesh(LoadParMesh(config, MPI_COMM_WORLD));
    pmesh->SetCurvature(order, true, dim, Ordering::byNODES);

    const bool is_coupled = (config.GetMaterialType() == "coupled");

    // --- Coupled path: evaluate fluid and solid submeshes separately ---
    // For coupled runs we split the parent mesh by domain attribute, build
    // the per-side material on its submesh, and call EvaluateMesh twice
    // with `<prefix>_fluid` / `<prefix>_solid` so histograms don't collide.
    // The CFL / PPW numbers on each side are what actually governs the
    // coupled time loop (dt is the min over BOTH submeshes), so a
    // per-side report is the useful diagnostic.
    if (is_coupled) {
        CoupledMaterialConfig coupled = LoadCoupledMaterialConfig(config);
        if (rank == 0) {
            std::cout << "  Coupled: fluid_attr=" << coupled.fluid_attribute
                      << " solid_attr=" << coupled.solid_attribute << "\n";
        }

        auto build_sub = [&](int attr) {
            Array<int> a(1); a[0] = attr;
            auto sub = std::make_unique<ParSubMesh>(
                ParSubMesh::CreateFromDomain(*pmesh, a));
            sub->SetCurvature(order, true, dim, Ordering::byNODES);
            return sub;
        };

        auto fluid_sub = build_sub(coupled.fluid_attribute);
        auto solid_sub = build_sub(coupled.solid_attribute);

        auto build_side_material = [&](const MaterialConfig& mc,
                                       ParMesh& sub)
            -> std::unique_ptr<MaterialBase>
        {
            const int side_dim = sub.Dimension();
            const std::string t = mc.material_type;
            H1_FECollection fec(order, side_dim);
            if (side_dim == 2) {
                auto* fes = new ParFiniteElementSpace(&sub, &fec);
                IntegrationRule ir_1d =
                    IntegrationRules(0, Quadrature1D::GaussLobatto)
                        .Get(Geometry::SEGMENT, 2 * order - 1);
                IntegrationRule ir(ir_1d, ir_1d);
                std::unique_ptr<MaterialBase> m;
                if (t == "isotropic_elastic") {
                    m = IsotropicElasticMaterial::FromConfig(mc, *fes, ir);
                } else {
                    m = IsotropicAcousticMaterial::FromConfig(mc, *fes, ir);
                }
                // Materials typically own or reference the fes; for an
                // evaluate-only tool we intentionally leak the fes so
                // the material's references stay valid for the lifetime
                // of the tool. (Not worth plumbing ownership for a
                // diagnostic.)
                return m;
            } else {
                auto* fes = new ParFiniteElementSpace(&sub, &fec);
                IntegrationRule ir_1d =
                    IntegrationRules(0, Quadrature1D::GaussLobatto)
                        .Get(Geometry::SEGMENT, 2 * order - 1);
                IntegrationRule ir(ir_1d, ir_1d, ir_1d);
                std::unique_ptr<MaterialBase> m;
                if (t == "isotropic_elastic") {
                    m = IsotropicElasticMaterial3D::FromConfig(mc, *fes, ir);
                } else {
                    m = IsotropicAcousticMaterial3D::FromConfig(mc, *fes, ir);
                }
                return m;
            }
        };

        auto fluid_mat = build_side_material(coupled.fluid, *fluid_sub);
        auto solid_mat = build_side_material(coupled.solid, *solid_sub);

        const auto side_prefix = [&](const std::string& side) {
            return histogram_prefix.empty()
                ? std::string("")
                : (histogram_prefix + "_" + side);
        };

        if (rank == 0) std::cout << "\n--- Fluid submesh ---\n";
        if (dim == 2) {
            EvaluateMesh<2>(*fluid_sub, *fluid_mat, order, max_freq,
                             ppw_target, cfl_factor, MPI_COMM_WORLD,
                             side_prefix("fluid"), nbins, config_file);
        } else {
            EvaluateMesh<3>(*fluid_sub, *fluid_mat, order, max_freq,
                             ppw_target, cfl_factor, MPI_COMM_WORLD,
                             side_prefix("fluid"), nbins, config_file);
        }

        if (rank == 0) std::cout << "\n--- Solid submesh ---\n";
        if (dim == 2) {
            EvaluateMesh<2>(*solid_sub, *solid_mat, order, max_freq,
                             ppw_target, cfl_factor, MPI_COMM_WORLD,
                             side_prefix("solid"), nbins, config_file);
        } else {
            EvaluateMesh<3>(*solid_sub, *solid_mat, order, max_freq,
                             ppw_target, cfl_factor, MPI_COMM_WORLD,
                             side_prefix("solid"), nbins, config_file);
        }

        if (rank == 0) std::cout << "\nDone.\n";
        Mpi::Finalize();
        return 0;
    }

    // --- Single-physics path (unchanged) ---
    if (rank == 0) std::cout << "Loading material..." << std::endl;
    std::unique_ptr<MaterialBase> material;
    if (dim == 2) {
        material = BuildMaterial<2>(config, *pmesh, order);
    } else {
        material = BuildMaterial<3>(config, *pmesh, order);
    }

    if (dim == 2) {
        EvaluateMesh<2>(*pmesh, *material, order, max_freq, ppw_target,
                         cfl_factor, MPI_COMM_WORLD, histogram_prefix, nbins,
                         config_file);
    } else {
        EvaluateMesh<3>(*pmesh, *material, order, max_freq, ppw_target,
                         cfl_factor, MPI_COMM_WORLD, histogram_prefix, nbins,
                         config_file);
    }

    if (rank == 0) std::cout << "\nDone.\n";

    Mpi::Finalize();
    return 0;
}
