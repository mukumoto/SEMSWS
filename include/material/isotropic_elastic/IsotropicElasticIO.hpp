/**
 * @file IsotropicElasticIO.hpp
 * @brief I/O functions for isotropic elastic material data
 *
 * Contains data structures and file I/O for isotropic elastic materials.
 * Part of the self-contained isotropic_elastic module.
 */

#ifndef SEM_ISOTROPIC_ELASTIC_IO_HPP
#define SEM_ISOTROPIC_ELASTIC_IO_HPP

#include <mfem.hpp>
#include <string>
#include <vector>

namespace SEM {

using namespace mfem;

// =============================================================================
// Grid Data Structures
// =============================================================================

/**
 * @brief Isotropic elastic material grid data for 2D simulations
 *
 * File format:
 * ```
 * # Comments start with #
 * nx ny
 * dx dy
 * x0 y0
 * vp vs rho [Qkappa Qmu]
 * ...
 * ```
 */
struct IsotropicElasticMaterialData2D {
    int nx, ny;              ///< Number of grid points
    real_t dx, dy;           ///< Grid spacing
    real_t x0, y0;           ///< Grid origin
    Vector vp;               ///< P-wave velocity [nx * ny]
    Vector vs;               ///< S-wave velocity [nx * ny]
    Vector rho;              ///< Density [nx * ny]
    Vector Qkappa;           ///< Q-factor for bulk modulus (optional)
    Vector Qmu;              ///< Q-factor for shear modulus (optional)
    bool has_Q = false;      ///< True if Qkappa/Qmu are available
};

/**
 * @brief Isotropic elastic material grid data for 3D simulations
 */
struct IsotropicElasticMaterialData3D {
    int nx, ny, nz;          ///< Number of grid points
    real_t dx, dy, dz;       ///< Grid spacing
    real_t x0, y0, z0;       ///< Grid origin
    Vector vp;               ///< P-wave velocity [nx * ny * nz]
    Vector vs;               ///< S-wave velocity [nx * ny * nz]
    Vector rho;              ///< Density [nx * ny * nz]
    Vector Qkappa;           ///< Q-factor for bulk modulus (optional)
    Vector Qmu;              ///< Q-factor for shear modulus (optional)
    bool has_Q = false;      ///< True if Qkappa/Qmu are available
};

// =============================================================================
// Attribute Entry Structure
// =============================================================================

/**
 * @brief Isotropic elastic material properties for a single mesh attribute
 */
struct IsotropicElasticAttributeEntry {
    int attribute;           ///< Mesh element attribute
    real_t vp;               ///< P-wave velocity
    real_t vs;               ///< S-wave velocity
    real_t rho;              ///< Density
    real_t Qkappa = -1.0;    ///< Q-factor for bulk modulus (-1 = unspecified)
    real_t Qmu = -1.0;       ///< Q-factor for shear modulus (-1 = unspecified)
};

// =============================================================================
// I/O Functions
// =============================================================================

/**
 * @brief Read isotropic elastic 2D ASCII material file
 * @param filename Path to ASCII file
 * @param[out] mat Output material data
 * @param read_Q If true, expect and read Qkappa/Qmu columns (file must have 5 columns)
 * @return true if successful
 */
bool ReadIsotropicElasticMaterialData2D(const std::string& filename,
                                        IsotropicElasticMaterialData2D& mat,
                                        bool read_Q = false);

/**
 * @brief Read isotropic elastic 3D ASCII material file
 * @param filename Path to ASCII file
 * @param[out] mat Output material data
 * @param read_Q If true, expect and read Qkappa/Qmu columns (file must have 5 columns)
 * @return true if successful
 */
bool ReadIsotropicElasticMaterialData3D(const std::string& filename,
                                        IsotropicElasticMaterialData3D& mat,
                                        bool read_Q = false);

/**
 * @brief Write isotropic elastic 2D ASCII material file
 * @param filename Path to output file
 * @param mat Material data to write
 * @param include_Q Whether to write Qkappa/Qmu columns
 * @return true if successful
 */
bool WriteIsotropicElasticMaterialData2D(const std::string& filename,
                                         const IsotropicElasticMaterialData2D& mat,
                                         bool include_Q = false);

/**
 * @brief Read isotropic elastic attribute-based material file
 *
 * File format:
 * ```
 * # attribute vp vs rho [Qkappa Qmu]
 * 1 3000.0 1732.0 2500.0
 * 2 2000.0 1000.0 2000.0
 * ```
 *
 * @param filename Path to attribute material file
 * @param[out] entries Output vector of attribute entries
 * @param read_Q If true, expect and read Qkappa/Qmu columns (file must have 6 columns)
 * @return true if successful
 */
bool ReadIsotropicElasticAttributeMaterial(const std::string& filename,
                                           std::vector<IsotropicElasticAttributeEntry>& entries,
                                           bool read_Q = false);

}  // namespace SEM

// Forward declarations to avoid pulling material headers into IO header.
namespace SEM {
class ElasticMaterialBase2D;
class ElasticMaterialBase3D;
}

namespace SEM {

/**
 * @brief Export 2D elastic material to ADIOS2 .bp files (vp, vs, rho [, qkappa, qmu])
 *
 * Reconstructs Vp = sqrt((kappa + 4/3 * mu) / rho), Vs = sqrt(mu / rho)
 * from the GLL-point fields stored in the elastic material.
 * Writes vp.bp, vs.bp, rho.bp (and qkappa.bp, qmu.bp when HasAttenuation()).
 *
 * @note When attenuation is enabled, the exported vp/vs reflect the
 *       unrelaxed-corrected kappa/mu that the material currently holds.
 *       For round-trip consistency, prefer attenuation OFF when using the
 *       exported BPs as input to a second run with format=adios2.
 *
 * @param material    Elastic material (must have Kappa(), Mu(), Rho())
 * @param export_dir  Directory to write output files
 * @param mesh_file   Mesh file path (stored as attribute)
 * @param comm        MPI communicator
 */
void ExportElasticMaterialBP(const ElasticMaterialBase2D& material,
                             const std::string& export_dir,
                             const std::string& mesh_file,
                             MPI_Comm comm);

/**
 * @brief Export 3D elastic material to ADIOS2 .bp files (vp, vs, rho [, qkappa, qmu])
 */
void ExportElasticMaterialBP(const ElasticMaterialBase3D& material,
                             const std::string& export_dir,
                             const std::string& mesh_file,
                             MPI_Comm comm);

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ELASTIC_IO_HPP
