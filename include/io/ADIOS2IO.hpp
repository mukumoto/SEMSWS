/**
 * @file ADIOS2IO.hpp
 * @brief ADIOS2-based I/O for MaterialField, L-BFGS history, and Adam state
 *
 * All ranks write to a single .bp file using ADIOS2 parallel I/O (BP5 engine).
 * Each rank writes its local elements as a contiguous block.
 *
 * Data layout per .bp file:
 *   Variable "data": shape {ne_global * nglly * ngllx} (1D flat)
 *   Each rank writes {ne_local * nglly * ngllx} starting at its offset
 *   Attributes: ne_global, ne_local, ngllx, nglly, mesh_file
 */

#ifndef SEM_ADIOS2IO_HPP
#define SEM_ADIOS2IO_HPP

#include "material/MaterialField.hpp"
#include <mfem.hpp>
#include <mpi.h>
#include <string>
#include <vector>

namespace SEM {

using namespace mfem;

/**
 * @brief Save a MaterialField to a .bp file (all ranks write in parallel)
 *
 * @param filename  Output .bp path
 * @param var_name  Variable name inside BP file (e.g., "data")
 * @param field     MaterialField to save (local partition)
 * @param mesh_file Path to mesh file (stored as attribute for later loading)
 * @param comm      MPI communicator
 */
void SaveFieldBP(const std::string& filename,
                 const std::string& var_name,
                 const MaterialField& field,
                 const std::string& mesh_file,
                 MPI_Comm comm);

/// Overload for 3D MaterialField. Writes the same BP schema with an
/// additional `ngllz` attribute and a per-element size of ngllz*nglly*ngllx.
/// sem_viz reads both the 2D and 3D variants through the same entry point,
/// dispatching on whether `ngllz` is present.
void SaveFieldBP(const std::string& filename,
                 const std::string& var_name,
                 const MaterialField3D& field,
                 const std::string& mesh_file,
                 MPI_Comm comm);

/**
 * @brief Load a MaterialField from a .bp file (all ranks read in parallel)
 *
 * Reads per-rank ne_local, ngllx, nglly from the .bp file metadata
 * to reconstruct the exact partition used when saving.
 * Must be called with the same number of MPI ranks as when saved.
 *
 * @param filename  Input .bp path
 * @param var_name  Variable name inside BP file (e.g., "data")
 * @param comm      MPI communicator
 * @return MaterialField with local partition data
 */
MaterialField LoadFieldBP(const std::string& filename,
                          const std::string& var_name,
                          MPI_Comm comm);

/// 3D counterpart — loads a BP file written with the 3D SaveFieldBP overload
/// (ngllz attribute present). Reads the per-rank partition via ne_local.
MaterialField3D LoadFieldBP3D(const std::string& filename,
                              const std::string& var_name,
                              MPI_Comm comm);

/**
 * @brief Save L-BFGS history vectors to a .bp file
 *
 * @param filename   Output .bp path
 * @param s          Step vectors s_k = x_{k+1} - x_k
 * @param y          Gradient difference vectors y_k = g_{k+1} - g_k
 * @param iteration  Current iteration number
 * @param comm       MPI communicator
 */
void SaveLBFGSHistoryBP(const std::string& filename,
                        const std::vector<Vector>& s,
                        const std::vector<Vector>& y,
                        int iteration, MPI_Comm comm);

/**
 * @brief L-BFGS history loaded from a .bp file
 */
struct LBFGSHistory {
    std::vector<Vector> s;  ///< Step vectors
    std::vector<Vector> y;  ///< Gradient difference vectors
    int iteration;          ///< Iteration number when saved
};

/**
 * @brief Load L-BFGS history from a .bp file
 *
 * @param filename  Input .bp path
 * @param comm      MPI communicator
 * @return LBFGSHistory with vectors and iteration number
 */
LBFGSHistory LoadLBFGSHistoryBP(const std::string& filename, MPI_Comm comm);

/**
 * @brief Adam optimizer state
 */
struct AdamState {
    Vector m;       ///< First moment estimate
    Vector v;       ///< Second moment estimate
    int iteration;  ///< Step count k (for bias correction: 1-β^k)
};

/**
 * @brief Save Adam optimizer state to a .bp file
 */
void SaveAdamStateBP(const std::string& filename,
                     const Vector& m, const Vector& v,
                     int iteration, MPI_Comm comm);

/**
 * @brief Load Adam optimizer state from a .bp file
 */
AdamState LoadAdamStateBP(const std::string& filename, MPI_Comm comm);

}  // namespace SEM

#endif  // SEM_ADIOS2IO_HPP
