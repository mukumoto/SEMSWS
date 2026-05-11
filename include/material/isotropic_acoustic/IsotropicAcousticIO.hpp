/**
 * @file IsotropicAcousticIO.hpp
 * @brief I/O functions for isotropic acoustic material data
 *
 * Contains data structures and file I/O for isotropic acoustic materials.
 * Part of the self-contained isotropic_acoustic module.
 */

#ifndef SEM_ISOTROPIC_ACOUSTIC_IO_HPP
#define SEM_ISOTROPIC_ACOUSTIC_IO_HPP

#include <mfem.hpp>
#include <string>
#include <vector>

namespace SEM {

using namespace mfem;

// =============================================================================
// Grid Data Structures
// =============================================================================

/**
 * @brief Isotropic acoustic material grid data for 2D simulations
 *
 * File format:
 * ```
 * # Comments start with #
 * nx ny
 * dx dy
 * x0 y0
 * vp rho [Qkappa]
 * ...
 * ```
 */
struct IsotropicAcousticMaterialData2D {
    int nx, ny;              ///< Number of grid points
    real_t dx, dy;           ///< Grid spacing
    real_t x0, y0;           ///< Grid origin
    Vector vp;               ///< P-wave velocity [nx * ny]
    Vector rho;              ///< Density [nx * ny]
    Vector Qkappa;           ///< Q-factor for bulk modulus (optional)
    bool has_Q = false;      ///< True if Qkappa is available
};

/**
 * @brief Isotropic acoustic material grid data for 3D simulations
 */
struct IsotropicAcousticMaterialData3D {
    int nx, ny, nz;          ///< Number of grid points
    real_t dx, dy, dz;       ///< Grid spacing
    real_t x0, y0, z0;       ///< Grid origin
    Vector vp;               ///< P-wave velocity [nx * ny * nz]
    Vector rho;              ///< Density [nx * ny * nz]
    Vector Qkappa;           ///< Q-factor for bulk modulus (optional)
    bool has_Q = false;      ///< True if Qkappa is available
};

// =============================================================================
// Attribute Entry Structure
// =============================================================================

/**
 * @brief Isotropic acoustic material properties for a single mesh attribute
 */
struct IsotropicAcousticAttributeEntry {
    int attribute;           ///< Mesh element attribute
    real_t vp;               ///< P-wave velocity
    real_t rho;              ///< Density
    real_t Qkappa = -1.0;    ///< Q-factor for bulk modulus (-1 = unspecified)
};

// =============================================================================
// I/O Functions
// =============================================================================

/**
 * @brief Read isotropic acoustic 2D ASCII material file
 * @param filename Path to ASCII file
 * @param[out] mat Output material data
 * @param read_Q If true, expect and read Qkappa column (file must have 3 columns)
 * @return true if successful
 */
bool ReadIsotropicAcousticMaterialData2D(const std::string& filename,
                                         IsotropicAcousticMaterialData2D& mat,
                                         bool read_Q = false);

/**
 * @brief Read isotropic acoustic 3D ASCII material file
 * @param filename Path to ASCII file
 * @param[out] mat Output material data
 * @param read_Q If true, expect and read Qkappa column (file must have 3 columns)
 * @return true if successful
 */
bool ReadIsotropicAcousticMaterialData3D(const std::string& filename,
                                         IsotropicAcousticMaterialData3D& mat,
                                         bool read_Q = false);

/**
 * @brief Read isotropic acoustic attribute-based material file
 *
 * File format:
 * ```
 * # attribute vp rho [Qkappa]
 * 1 1500.0 1000.0
 * 2 2000.0 1200.0
 * ```
 *
 * @param filename Path to attribute material file
 * @param[out] entries Output vector of attribute entries
 * @param read_Q If true, expect and read Qkappa column (file must have 4 columns)
 * @return true if successful
 */
bool ReadIsotropicAcousticAttributeMaterial(const std::string& filename,
                                            std::vector<IsotropicAcousticAttributeEntry>& entries,
                                            bool read_Q = false);

// =============================================================================
// ADIOS2 Export Functions
// =============================================================================

class AcousticMaterialBase2D;
class AcousticMaterialBase3D;

/**
 * @brief Export 2D acoustic material to ADIOS2 .bp files (vp, rho)
 *
 * Converts kappa/inv_rho back to physical properties:
 *   rho = 1 / inv_rho
 *   vp  = sqrt(kappa * inv_rho)
 *
 * @param material  Acoustic material (must have Kappa() and InvRho())
 * @param export_dir  Directory to write output files
 * @param mesh_file   Mesh file path (stored as attribute)
 * @param comm        MPI communicator
 */
void ExportAcousticMaterialBP(const AcousticMaterialBase2D& material,
                              const std::string& export_dir,
                              const std::string& mesh_file,
                              MPI_Comm comm);

/**
 * @brief Export 3D acoustic material to ADIOS2 .bp files (vp, rho)
 */
void ExportAcousticMaterialBP(const AcousticMaterialBase3D& material,
                              const std::string& export_dir,
                              const std::string& mesh_file,
                              MPI_Comm comm);

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ACOUSTIC_IO_HPP
