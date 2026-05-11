/**
 * @file ConfigLoaders.hpp
 * @brief Functions to load simulation components from YamlConfig
 *
 * This file provides loader functions that bridge YAML configuration
 * to SEM component objects (sources, receivers, materials, mesh).
 *
 * Supported file formats:
 * - Sources: Inline YAML or external YAML file (shot_list.yaml format)
 * - Receivers: Inline YAML, external text file, or receiver grid
 * - Material: Constant, HDF5, or ASCII formats
 * - Mesh: Internal Cartesian or external MFEM/Gmsh formats
 */

#ifndef SEM_CONFIG_LOADERS_HPP
#define SEM_CONFIG_LOADERS_HPP

#include "config/YamlConfig.hpp"
#include "config/ConfigTypes.hpp"
#include "common/BoundaryUtils.hpp"
#include <mfem.hpp>

namespace SEM {


using namespace mfem;

// =============================================================================
// Configuration Reading (Pure data extraction, no object creation)
// =============================================================================

/**
 * @brief Load material configuration from YAML
 *
 * Extracts material parameters from YAML config into a MaterialConfig struct.
 * Does NOT create material objects - use Material::FromConfig() for that.
 *
 * @param config YAML configuration
 * @return MaterialConfig with extracted values
 */
MaterialConfig LoadMaterialConfig(const YamlConfig& config);

/**
 * @brief Load fluid-solid coupled material configuration from YAML.
 *
 * Requires `material.type: coupled`. Extracts the nested `material.fluid.*`
 * and `material.solid.*` blocks into a CoupledMaterialConfig. The interface
 * boundary attribute is NOT read from YAML — it is auto-detected at run
 * time in FluidSolidInterface::Setup() via set-difference on ParSubMesh
 * bdr_attributes.
 *
 * @param config YAML configuration (must have IsCoupledMaterial()==true)
 * @return CoupledMaterialConfig with fluid+solid sub-configs and attributes
 */
CoupledMaterialConfig LoadCoupledMaterialConfig(const YamlConfig& config);

/**
 * @brief Load mixed attribute material entries from external YAML file
 *
 * Reads the external YAML file specified in by_attribute_file and populates
 * the attribute_entries vector in MaterialConfig. Used for format="by_attribute_mixed".
 *
 * @param yaml_path Path to the external YAML file (e.g., materials/mixed.yaml)
 * @param entries Output vector of AttributeMaterialEntry
 *
 * Expected YAML format:
 *   attributes:
 *     1:
 *       mode: constant
 *       vp: 3000.0
 *       vs: 1732.0
 *       rho: 2500.0
 *     2:
 *       mode: grid
 *       file: materials/hetero.txt  # Relative to this YAML file
 */
void LoadAttributeMixedFromYaml(
    const std::string& yaml_path,
    std::vector<AttributeMaterialEntry>& entries);

/**
 * @brief Load 2D source configuration from YAML
 *
 * Extracts source parameters from YAML config into a SourceConfig::Config2D struct.
 * Does NOT create source objects - use PointSourceCollection::FromConfig() for that.
 *
 * @param config YAML configuration
 * @return SourceConfig::Config2D with extracted values
 */
SourceConfig::Config2D LoadSourceConfig2D(const YamlConfig& config);

/**
 * @brief Load 3D source configuration from YAML
 *
 * Extracts source parameters from YAML config into a SourceConfig::Config3D struct.
 * Does NOT create source objects - use PointSourceCollection::FromConfig() for that.
 *
 * @param config YAML configuration
 * @return SourceConfig::Config3D with extracted values
 */
SourceConfig::Config3D LoadSourceConfig3D(const YamlConfig& config);


// =============================================================================
// Mesh Loading
// =============================================================================

/**
 * @brief Create mesh from YAML configuration
 * @param config YAML configuration
 * @return Pointer to serial mesh (caller owns memory)
 *
 * Supports:
 * - type: internal - Creates Cartesian mesh from origin/size/elements
 * - type: external - Loads from file (mfem, gmsh, vtk, exodus)
 */
Mesh* LoadMesh(const YamlConfig& config);

/**
 * @brief Create parallel mesh from YAML configuration
 * @param config YAML configuration
 * @param comm MPI communicator
 * @return Pointer to parallel mesh (caller owns memory)
 *
 * Supports two modes:
 * - type: internal/external: Rank 0 loads mesh, partitions, and distributes via MPI
 * - type: partitioned: Each rank loads its own partition file (most memory efficient)
 */
ParMesh* LoadParMesh(const YamlConfig& config, MPI_Comm comm);

/**
 * @brief Load parallel mesh from pre-partitioned files
 * @param config YAML configuration with mesh.type="partitioned"
 * @param comm MPI communicator
 * @return Pointer to parallel mesh (caller owns memory)
 *
 * Each MPI rank reads ONLY its own partition file. This is the most
 * memory-efficient loading method for large-scale simulations.
 *
 * Required config:
 *   mesh.type: partitioned
 *   mesh.partitioned.directory: path to partition files
 *   mesh.partitioned.nparts: number of partitions (must equal nprocs)
 *   mesh.partitioned.pattern: filename pattern (default: "mesh.%06d.mesh")
 *
 * File naming: {directory}/{pattern} where pattern uses printf format
 */
ParMesh* LoadParMeshFromPartition(const YamlConfig& config, MPI_Comm comm);


// =============================================================================
// Utility Functions
// =============================================================================

// ParseBoundarySide is now in common/BoundaryUtils.hpp

/**
 * @brief Get list of boundary attributes from ABC config
 * @param abc ABC configuration
 * @param dim Dimension
 * @return Array of boundary attributes
 */
Array<int> GetABCBoundaryAttributes(const ABCConfig& abc, int dim);

/**
 * @brief Get Dirichlet boundary attributes from YAML config
 *
 * Supports both named sides ("top", "left") and integer attribute indices ("5").
 *
 * @param config YAML configuration
 * @param dim Dimension (2 or 3)
 * @return Array of boundary attributes for Dirichlet BC
 */
Array<int> GetDirichletBoundaryAttributes(const YamlConfig& config, int dim);

/**
 * @brief Get essential (Dirichlet) true DOFs from boundary attributes
 *
 * Identifies all DOFs on the specified boundaries for Dirichlet BC enforcement.
 *
 * @param fes Finite element space
 * @param dirichlet_attrs Boundary attributes for Dirichlet BC
 * @return Array of true DOF indices for Dirichlet BC
 */
Array<int> GetDirichletTrueDofs(ParFiniteElementSpace& fes,
                                const Array<int>& dirichlet_attrs);

// =============================================================================
// Note: Material I/O functions are now in each material's own IO module:
// - isotropic_elastic: material/isotropic_elastic/IsotropicElasticIO.hpp
// - isotropic_acoustic: material/isotropic_acoustic/IsotropicAcousticIO.hpp
// =============================================================================

}  // namespace SEM

#endif  // SEM_CONFIG_LOADERS_HPP
