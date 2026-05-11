/**
 * @file combine_mesh.cpp
 * @brief MPI tool to combine partitioned mesh files into a single mesh for GLVis
 *
 * This tool loads pre-partitioned mesh files (created by partition_mesh) and
 * saves them as a single combined mesh using ParMesh::SaveAsOne(). The output
 * is compatible with ParGridFunction::SaveAsOne() for GLVis visualization.
 *
 * Usage:
 *   mpirun -np 64 ./combine_mesh <partition_dir> <output_mesh>
 *   mpirun -np 64 ./combine_mesh --device cuda <partition_dir> <output_mesh>
 *
 * IMPORTANT: Must run with the same number of MPI ranks as partitions!
 */

#include <mfem.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace mfem;

void PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] <partition_dir> <output_mesh>\n\n"
              << "Combines partitioned mesh files into a single mesh for GLVis visualization.\n"
              << "IMPORTANT: Must run with same number of MPI ranks as partitions!\n\n"
              << "Options:\n"
              << "  -h, --help           Show this help message\n"
              << "  --device cpu|cuda    Device backend (default: cpu)\n\n"
              << "Input files expected: mesh.mesh.000000, mesh.mesh.000001, ...\n\n"
              << "Examples:\n"
              << "  # Combine 64 partitions (must use mpirun -np 64)\n"
              << "  mpirun -np 64 " << prog << " partitions/ combined.mesh\n";
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Positional arguments
    std::string partition_dir;
    std::string output_file;
    std::string device_str = "cpu";

    // Parse command line
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            if (rank == 0) PrintUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else if (arg == "--device" && i + 1 < argc) {
            device_str = argv[++i];
        } else if (arg[0] != '-') {
            if (partition_dir.empty()) {
                partition_dir = arg;
            } else {
                output_file = arg;
            }
        } else {
            if (rank == 0) {
                std::cerr << "Unknown option: " << arg << std::endl;
                PrintUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Validate arguments
    if (partition_dir.empty() || output_file.empty()) {
        if (rank == 0) {
            std::cerr << "Error: partition_dir and output_mesh are required\n";
            PrintUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    Device device(device_str);

    // Construct filename for this rank (e.g., mesh.mesh.000000)
    std::ostringstream oss;
    oss << partition_dir << "/mesh.mesh." << std::setfill('0') << std::setw(6) << rank;
    std::string filename = oss.str();

    // Load partition file
    if (rank == 0) {
        std::cout << "Loading " << nprocs << " partition files from: " << partition_dir << std::endl;
    }

    std::ifstream ifs(filename);
    if (!ifs.good()) {
        std::cerr << "Error: Cannot open partition file: " << filename
                  << " (rank " << rank << ")" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    ParMesh pmesh(MPI_COMM_WORLD, ifs);
    ifs.close();

    // Report mesh info
    int local_ne = pmesh.GetNE();
    int total_ne;
    MPI_Reduce(&local_ne, &total_ne, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Total elements: " << total_ne << ", dimension: " << pmesh.Dimension() << std::endl;
        std::cout << "Saving combined mesh to: " << output_file << std::endl;
    }

    // Save combined mesh using SaveAsOne
    pmesh.SaveAsOne(output_file);

    if (rank == 0) {
        std::cout << "Done! Combined mesh saved to: " << output_file << std::endl;
        std::cout << "\nFor GLVis visualization:\n";
        std::cout << "  glvis -m " << output_file << " -g <solution.gf>\n";
        std::cout << "\nNote: The solution.gf must be saved with ParGridFunction::SaveAsOne()\n";
    }

    MPI_Finalize();
    return 0;
}
