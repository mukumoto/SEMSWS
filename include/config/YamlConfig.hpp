/**
 * @file YamlConfig.hpp
 * @brief YAML configuration file parser for SEMSWS
 *
 * Provides parsing of YAML configuration files as specified in
 * SOFTWARE_SPECIFICATION.md Section 5.2.
 *
 * Design principles:
 * - Simple, explicit typed methods (no templates)
 * - Nested struct types for complex configuration groups
 * - Direct YAML node access using yaml-cpp API
 */

#ifndef SEM_YAML_CONFIG_HPP
#define SEM_YAML_CONFIG_HPP

#include <mfem.hpp>
#include <string>
#include <vector>
#include "config/ConfigTypes.hpp"
#include "common/BoundaryUtils.hpp"
#include <yaml-cpp/yaml.h>

namespace SEM {

using mfem::real_t;

// =============================================================================
// Configuration Structures
// =============================================================================

/**
 * @brief Absorbing boundary condition configuration
 */
struct ABCConfig {
    std::string type;                    // "kosloff" | "shin" | "pml"
    std::vector<std::string> sides;      // ["left", "right", "bottom", ...]
    real_t thickness;                    // Sponge layer thickness (m)
    real_t alpha;                        // Damping coefficient (-1 for auto)

    ABCConfig() : type("kosloff"), thickness(0.0), alpha(-1.0) {}
};

/**
 * @brief Optional resampling spec for observed data
 */
struct ObservedResampleDef {
    bool enabled = false;
    std::string method = "lanczos";   // "lanczos" | "linear"
    int lanczos_a = 4;
};

/**
 * @brief Observed data definition (per-source).
 *
 * One source = one HDF5 file (canonical format). All receiver metadata
 * (positions, channels, weights, dt/t0/n_samples/space_dim) live inside
 * the HDF5 file itself.
 */
struct ObservedSourceDef {
    std::string file;                    // path to source_NNNN.h5
    ObservedResampleDef resample;
};

/**
 * @brief Source definition from YAML
 */
struct SourceDef {
    int id;                              // Source identifier
    std::string name;                    // Source name
    std::string type;                    // "force" | "moment_tensor" | "pressure"
    real_t location[3];                  // [x, y, z] position
    real_t direction[3];                 // [dx, dy, dz] unit vector (for force)

    // Wavelet parameters
    std::string wavelet_type;            // "ricker" | "gaussian" | "external" | "hdf5"
    real_t frequency;                    // Dominant frequency (Hz)
    real_t amplitude;                    // Amplitude scaling
    real_t delay;                        // Time delay (s)
    std::string external_file;           // Path to external STF file (for wavelet_type="external")
    std::vector<real_t> stf_samples;     // Pre-loaded STF samples (wavelet_type="hdf5"); length == nt

    // Moment tensor components (for moment_tensor type)
    real_t M[6];                         // Mxx, Myy, Mzz, Mxy, Mxz, Myz

    // Observed data (inversion mode only)
    ObservedSourceDef observed;          // Canonical HDF5 observed-data spec
    bool has_observed;                   // Whether observed section exists

    SourceDef() : id(0), type("force"), wavelet_type("ricker"),
                  frequency(0.0), amplitude(1.0), delay(0.0),
                  has_observed(false) {
        for (int i = 0; i < 3; i++) {
            location[i] = 0.0;
            direction[i] = 0.0;
        }
        direction[1] = 1.0;  // Default: vertical force
        for (int i = 0; i < 6; i++) M[i] = 0.0;
    }
};

/**
 * @brief Receiver definition from YAML
 */
struct ReceiverDef {
    std::string name;                    // Receiver name/identifier
    real_t location[3];                  // [x, y, z] position
    std::vector<std::string> types;      // ["velocity", "displacement", ...] multiple types
    real_t weight;                       // Optional weight (default 1.0)

    ReceiverDef() : weight(1.0) {
        for (int i = 0; i < 3; i++) location[i] = 0.0;
    }
};

/**
 * @brief Receiver line definition for linear spacing between two points
 */
struct ReceiverLineDef {
    real_t start[3];                     // Start position
    real_t end_[3];                      // End position (end_ to avoid keyword)
    int count;                           // Number of receivers
    std::string prefix;                  // Name prefix

    ReceiverLineDef() : count(0), prefix("REC") {
        for (int i = 0; i < 3; i++) {
            start[i] = 0.0;
            end_[i] = 0.0;
        }
    }
};


// =============================================================================
// YamlConfig Class
// =============================================================================

/**
 * @brief YAML configuration file parser
 *
 * Parses YAML configuration files according to SOFTWARE_SPECIFICATION.md
 * Section 5.2 format. Provides simple accessor methods for all configuration
 * parameters.
 *
 * Example usage:
 * @code
 *   SEM::YamlConfig config("simulation.yaml");
 *   if (!config.IsValid()) {
 *       std::cerr << config.GetValidationError() << std::endl;
 *       return 1;
 *   }
 *   std::string mat_type = config.GetMaterialType();
 *   int order = config.GetOrder();
 * @endcode
 */
class YamlConfig {
public:
    /**
     * @brief Construct from YAML file
     * @param filepath Path to YAML configuration file
     */
    explicit YamlConfig(const std::string& filepath);

    /**
     * @brief Default constructor (creates invalid config)
     */
    YamlConfig();

    // =========================================================================
    // Validation
    // =========================================================================

    /** @brief Check if configuration is valid */
    bool IsValid() const { return valid_; }

    /** @brief Get validation error message */
    std::string GetValidationError() const { return error_msg_; }


    // =========================================================================
    // Simulation Section
    // =========================================================================

    /** @brief Get simulation name/identifier */
    std::string GetName() const;

    // GetPhysics() removed - physics type is now inferred from MaterialType
    // GetBackend() removed - unused

    /** @brief Get spatial dimension: 2 or 3 */
    int GetDimension() const;

    /** @brief Get polynomial order */
    int GetOrder() const;

    /** @brief Get number of time steps */
    int GetNumSteps() const;

    /** @brief Get time step (s) */
    real_t GetDt() const;

    /** @brief Get simulation start time t0 (s), default 0.0 */
    real_t GetT0() const;

    /** @brief Get CFL safety factor */
    real_t GetCflFactor() const;

    /** @brief Get output directory */
    std::string GetOutputDirectory() const;

    // GetSeismogramFormat() removed - unused

    /** @brief Check if wavefield output is enabled */
    bool IsWavefieldOutputEnabled() const;

    /** @brief Get wavefield output interval (steps) */
    int GetWavefieldInterval() const;

    /** @brief Get full wavefield output configuration (multi-format) */
    WavefieldOutputConfig GetWavefieldOutputConfig() const;

    /**
     * @brief Get wavefield output configuration with per-side override.
     *
     * For coupled runs the YAML may declare a per-domain override under
     * `simulation.output.wavefield.<side>` (side = "fluid" | "solid").
     * Any key (interval, fields, formats, format) set in that sub-block
     * overrides the corresponding value from the top-level wavefield
     * block. Keys left unset inherit from the top-level defaults, so a
     * config that only needs sparser solid snapshots can write just
     * `solid: { interval: 2000 }`.
     *
     * If the override sub-block is absent or empty, the returned config
     * is identical to `GetWavefieldOutputConfig()`.
     */
    WavefieldOutputConfig GetWavefieldOutputConfig(
        const std::string& side) const;

    /** @brief Get material output configuration */
    MaterialOutputConfig GetMaterialOutputConfig() const;

    /**
     * @brief Get material output configuration with per-side override.
     *
     * Mirrors the per-side wavefield accessor: for coupled runs, YAMLs
     * may declare `simulation.output.material.<side>` (side =
     * "fluid" | "solid") whose keys (enabled, fields, formats)
     * override the corresponding top-level defaults. Unset keys in the
     * override inherit the top-level values. If the override block is
     * absent this returns the same config as the no-arg overload.
     */
    MaterialOutputConfig GetMaterialOutputConfig(
        const std::string& side) const;

    /** @brief Check if mesh saving is enabled (mesh.save) */
    bool GetMeshSave() const;

    /** @brief Get log output interval (steps) */
    int GetLogInterval() const;

    /** @brief Get summary file path (empty if not specified) */
    std::string GetSummaryFile() const;

    // =========================================================================
    // Mesh Section
    // =========================================================================

    /** @brief Get maximum frequency for mesh evaluation (mesh.max_freq, required) */
    real_t GetMaxFreq() const;

    /** @brief Get required PPW for mesh evaluation (mesh.ppw, required) */
    real_t GetPPW() const;

    /** @brief Get mesh type: "external" | "internal" */
    std::string GetMeshType() const;

    /** @brief Get external mesh file path */
    std::string GetMeshFile() const;

    /** @brief Get mesh format: "gmsh" | "mfem" | "exodus" | "vtk" */
    std::string GetMeshFormat() const;

    /**
     * @brief Get internal mesh origin
     * @param origin Array to fill [x, y] (2D) or [x, y, z] (3D)
     */
    void GetMeshOrigin(real_t* origin) const;

    /**
     * @brief Get internal mesh size
     * @param size Array to fill [lx, ly] (2D) or [lx, ly, lz] (3D)
     */
    void GetMeshSize(real_t* size) const;

    /**
     * @brief Get internal mesh element counts
     * @param nel Array to fill [nx, ny] (2D) or [nx, ny, nz] (3D)
     */
    void GetMeshElements(int* nel) const;

    /** @brief y-coordinate threshold for splitting internal mesh into 2 attributes
     *  Returns NaN when not set. Elements with center.y > threshold get
     *  attribute=1; the rest get attribute=2. Used to mark a water layer
     *  (attr 1, frozen) vs an inverted solid (attr 2). */
    real_t GetMeshAttrYThreshold() const;

    /** @brief Get mesh partitioning method: "metis" | "cartesian" */
    std::string GetMeshPartition() const;

    /** @brief Get partition grid for Cartesian partitioning */
    void GetPartitionGrid(int* nxyz) const;

    // -------------------------------------------------------------------------
    // Partitioned Mesh (pre-partitioned files for memory-efficient loading)
    // -------------------------------------------------------------------------

    /** @brief Check if using pre-partitioned mesh files (mesh.type == "partitioned") */
    bool UsePartitionedMesh() const;

    /** @brief Get directory containing partition files */
    std::string GetPartitionDirectory() const;

    /** @brief Get number of partitions (must equal nprocs at runtime) */
    int GetPartitionCount() const;

    // =========================================================================
    // Material Section
    // =========================================================================

    /** @brief Get material type: "isotropic" | "vti" | "tti" */
    std::string GetMaterialType() const;

    /** @brief Get material format: "hdf5" | "ascii" | "constant" */
    std::string GetMaterialFormat() const;

    /**
     * @brief Whether this YAML has `material.type: coupled` (fluid-solid).
     *
     * When true, the flat material getters (GetMaterialFormat, vp/rho etc.)
     * must NOT be called; use GetCoupledMaterialConfig() instead.
     */
    bool IsCoupledMaterial() const;

    /**
     * @brief Parse the nested fluid/solid sub-materials under `material:`.
     *
     * Expected structure:
     *   material:
     *     type: coupled
     *     fluid:
     *       attribute: <int>
     *       type: isotropic_acoustic
     *       format: constant | grid | ...
     *       ...                 # same keys as a flat acoustic material block
     *     solid:
     *       attribute: <int>
     *       type: isotropic_elastic | anisotropic_elastic
     *       ...
     *
     * The interface boundary attribute is NOT read here; it is detected at
     * run time in FluidSolidInterface::Setup() via set-difference on the
     * submesh bdr_attributes.
     */
    CoupledMaterialConfig GetCoupledMaterialConfig() const;

    /** @brief Get material file path */
    std::string GetMaterialFile() const;


    /**
     * @brief Get constant material properties
     * @param vp P-wave velocity (m/s)
     * @param vs S-wave velocity (m/s)
     * @param rho Density (kg/m³)
     */
    void GetConstantMaterialVpVsRho(real_t* vp, real_t* vs, real_t* rho) const;

    void GetConstantMaterialVpRho(real_t* vp, real_t* rho) const;

    /** @brief Check if attenuation is enabled */
    bool IsAttenuationEnabled() const;

    /** @brief Get attenuation reference frequency (Hz) */
    real_t GetAttenuationF0() const;

    /** @brief Get number of relaxation mechanisms for attenuation */
    int GetAttenuationNumUnits() const;

    /** @brief Get constant Q_kappa value (returns -1 if from file) */
    real_t GetConstantQkappa() const;

    /** @brief Get constant Q_mu value (returns -1 if from file) */
    real_t GetConstantQmu() const;

    /** @brief Get Q_kappa file path (empty if constant) */
    std::string GetQkappaFile() const;

    /** @brief Get Q_mu file path (empty if constant) */
    std::string GetQmuFile() const;

    // GetQkappaDataset(), GetQmuDataset() removed - unused
    // GetAsciiMaterialParams() removed - unused (AsciiMaterialParams struct also removed)
    // HasMaterialByAttribute(), GetMaterialByAttribute() removed - unused (AttributeMaterialDef struct also removed)

    /**
     * @brief Get path to by_attribute material file
     *
     * Used when material.format is "by_attribute".
     * @return File path to attribute material file
     */
    std::string GetMaterialByAttributeFile() const;

    /** @brief Get ADIOS2 Vp .bp file path (for format="adios2") */
    std::string GetADIOS2VpFile() const;

    /** @brief Get ADIOS2 Vs .bp file path (for format="adios2", elastic only) */
    std::string GetADIOS2VsFile() const;

    /** @brief Get ADIOS2 rho .bp file path (for format="adios2") */
    std::string GetADIOS2RhoFile() const;

    /** @brief Get ADIOS2 Qkappa .bp file path (for format="adios2", visco) */
    std::string GetADIOS2QkappaFile() const;

    /** @brief Get ADIOS2 Qmu .bp file path (for format="adios2", visco-elastic) */
    std::string GetADIOS2QmuFile() const;

    /** @brief Check if model export is enabled (material.export_model: true) */
    bool IsExportModelEnabled() const;

    /** @brief Get model export directory (material.export_dir, default "./model/") */
    std::string GetExportModelDir() const;

    // =========================================================================
    // Boundary Section
    // =========================================================================

    /** @brief Get absorbing boundary configuration */
    ABCConfig GetABCConfig() const;

    /** @brief Get Dirichlet boundary attribute values (named sides or integers) */
    std::vector<std::string> GetDirichletAttributes() const;

    // =========================================================================
    // Sources Section
    // =========================================================================

    /** @brief Get source execution mode: "simultaneous" | "sequential" */
    std::string GetSourceMode() const;

    /** @brief Get external source file path (empty if inline) */
    std::string GetSourceFile() const;

    /** @brief Source input format: "yaml" (default; inline list / external
     *         YAML) or "hdf5" (v2.0 SEMSWS HDF5 file). */
    std::string GetSourceFormat() const;

    /** @brief HDF5 shot index (only meaningful when GetSourceFormat()=="hdf5").
     *         Default 0. */
    int GetSourceShotId() const;

    /** @brief Get all source definitions */
    std::vector<SourceDef> GetAllSources() const;

    // =========================================================================
    // Receivers Section
    // =========================================================================

    /** @brief Check if receivers section exists */
    bool HasReceivers() const;

    /** @brief Get external receiver file path (empty if inline) */
    std::string GetReceiverFile() const;

    /** @brief Receiver input format: "yaml" (default; inline list / line /
     *         external YAML) or "hdf5" (v2.0 SEMSWS HDF5 file). */
    std::string GetReceiverFormat() const;

    /** @brief HDF5 shot index (only meaningful when GetReceiverFormat()=="hdf5").
     *         Default 0. */
    int GetReceiverShotId() const;

    /** @brief Get all receiver definitions */
    std::vector<ReceiverDef> GetAllReceivers() const;

    /** @brief Check if receiver line is defined */
    bool HasReceiverLine() const;

    /** @brief Get receiver line definition */
    ReceiverLineDef GetReceiverLine() const;

    /** @brief Get receiver output formats (one or more of: "hdf5", "ascii", "su") */
    std::vector<std::string> GetReceiverOutputFormats() const;

    /** @brief Get receiver output filename */
    std::string GetReceiverOutputFilename() const;

    // =========================================================================
    // Device Section (replaces Parallel Section)
    // =========================================================================

    /** @brief Get device type: "cpu" | "cuda" | "hip" */
    std::string GetDevice() const;

    /** @brief Get seismogram buffer steps for GPU recording (0 = all steps) */
    int GetSeismoBufferSteps() const;

    // =========================================================================
    // Simulation Mode
    // =========================================================================

    /** @brief Get simulation mode: "forward" (default) | "inversion" | "misfit_only" */
    std::string GetSimulationMode() const;

    // =========================================================================
    // Inversion Parameters (simulation direct, used when mode == "inversion")
    // =========================================================================

    /** @brief Get misfit type: "l2_waveform" (default) or
     *         "normalized_correlation". */
    std::string GetMisfitType() const;

    /** @brief Get number of Revolve checkpoints */
    int GetNumCheckpoints() const;

    /** @brief Get checkpoint storage type: "memory" (default) | "disk" */
    std::string GetCheckpointStorage() const;

    /** @brief Get checkpoint device placement: "auto" (default) | "host" | "device" */
    std::string GetCheckpointDevice() const;

    /** @brief Get checkpoint directory (for disk storage) */
    std::string GetCheckpointDir() const;

    /** @brief Get kernel output directory */
    std::string GetKernelOutputDir() const;

    /**
     * @brief Get the sensitivity-kernel backend selector.
     *
     * Reads `inversion.sensitivity.backend` from the YAML config.
     * Valid values: "hand" (default, production hand-derived chain rule)
     *               "ad"   (forward-mode AD via mfem::future::dual)
     *
     * AD backend is currently wired only for 2D acoustic
     * (IsotropicAcoustic). See test/unit/acoustic_ad_vs_hand_test.cpp for
     * the numerical parity verification.
     */
    std::string GetSensitivityBackend() const;

    /// Whether Q (Qκ, Qμ) should be treated as an inversion parameter.
    /// When true, the sensitivity kernel factory must return a backend
    /// that accumulates K_Qκ / K_Qμ alongside K_Vp / K_Vs / K_ρ; the `hand`
    /// backend does not implement this and the factory aborts. Reads
    /// `inversion.invert_Q: bool` (default false).
    bool GetInvertQ() const;

private:
    YAML::Node root_;                    // Root YAML node
    std::string filepath_;               // Path to config file
    bool valid_;                         // Validation status
    std::string error_msg_;              // Error message if invalid

    // Cache for parsed sources/receivers (computed once)
    mutable std::vector<SourceDef> sources_cache_;
    mutable std::vector<ReceiverDef> receivers_cache_;
    mutable bool sources_parsed_;
    mutable bool receivers_parsed_;

    /** @brief Validate configuration */
    void Validate();

    /** @brief Parse sources from YAML */
    void ParseSources() const;

    /** @brief Parse receivers from YAML */
    void ParseReceivers() const;

    // =========================================================================
    // Helper methods (no templates, explicit types, no defaults)
    // =========================================================================

    /** @brief Read string from YAML node (MFEM_ABORT if not found) */
    std::string ReadString(const YAML::Node& node, const std::string& key) const;

    /** @brief Read real_t from YAML node (MFEM_ABORT if not found) */
    real_t Readreal_t(const YAML::Node& node, const std::string& key) const;

    /** @brief Read int from YAML node (MFEM_ABORT if not found) */
    int ReadInt(const YAML::Node& node, const std::string& key) const;

    /** @brief Read bool from YAML node (MFEM_ABORT if not found) */
    bool ReadBool(const YAML::Node& node, const std::string& key) const;

    /** @brief Read real_t array from YAML node (MFEM_ABORT if not found) */
    void ReadRealArray(const YAML::Node& node, const std::string& key,
                       real_t* arr, int size) const;

    /** @brief Read int array from YAML node (MFEM_ABORT if not found) */
    void ReadIntArray(const YAML::Node& node, const std::string& key,
                      int* arr, int size) const;

    /** @brief Read string array from YAML node (returns empty if not found) */
    std::vector<std::string> ReadStringArray(const YAML::Node& node,
                                              const std::string& key) const;
};

}  // namespace SEM

#endif  // SEM_YAML_CONFIG_HPP
