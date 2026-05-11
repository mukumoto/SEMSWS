// Loads source/receiver geometry from a v2.0 SEMSWS HDF5 file
// (see include/srcrecv/HDF5IOSchema.hpp). Forward-input use case
// (positions/types only, no waveform payload). For the FWI observed-data
// use case use HDF5ObservedReader instead.
//
// Schema slice consumed:
//   /                                  attrs: format_version, dt, t0,
//                                              n_samples, space_dim, ...
//   /shots/<NNNN>/receivers/<key>/     attrs: position[space_dim]
//                                              (optional) types[]
//                                              (optional) label
//
// Per-receiver overrides:
//   @types  (variable-length string array): overrides the parent YAML
//           `receivers.type:` for this receiver only.
//   @label  (str): display name; falls back to the group key when absent.

#ifndef SEM_HDF5_SOURCE_RECEIVER_READER_HPP
#define SEM_HDF5_SOURCE_RECEIVER_READER_HPP

#include "config/ConfigTypes.hpp"
#include "common/Types.hpp"

#include <string>
#include <vector>

namespace SEM {

/// One source entry read from `/shots/<NNNN>/sources/S<id>/`.
/// Covers `force`, `pressure`, and `moment_tensor` types.
struct HDF5SourceEntry {
    int id = 0;                          ///< @id (positive integer, unique per shot)
    std::string label;                   ///< @label (optional, may be empty)
    std::string type;                    ///< "force" | "pressure" | "moment_tensor"
    std::vector<real_t> position;        ///< size == space_dim
    std::vector<real_t> direction;       ///< size == space_dim, populated for "force"
    std::vector<real_t> stf;             ///< size == n_samples (scalar STF, F2)
    /// Moment tensor components, in canonical SEMSWS order (already remapped
    /// from the file's `@component_order`). 3D: 6 entries
    /// {Mxx,Myy,Mzz,Mxy,Mxz,Myz}; 2D: 3 entries {Mxx,Myy,Mxy}. Empty for
    /// non-MT types.
    std::vector<real_t> moment_tensor;
};

struct HDF5SourceCatalog {
    int n_samples = 0;
    real_t dt = 0.0;
    real_t t0 = 0.0;
    int space_dim = 0;
    std::vector<HDF5SourceEntry> sources;
};

class HDF5SourceReceiverReader {
public:
    /**
     * @brief Read receiver geometry from `path` under `/shots/<shot_id>/receivers/`.
     *
     * @param path Path to the v2.0 SEMSWS HDF5 file.
     * @param shot_id Shot index (default 0). Reader fails if the group is missing.
     * @param default_types Receiver type list to apply when an entry has no
     *        per-receiver `@types` attribute. Comes from YAML `receivers.type:`.
     * @param expected_space_dim Asserted against the file's root `space_dim`.
     * @return ReceiverConfig::Config with `receivers` populated. The
     *         `output_format` field is left at its default; callers fill
     *         it from YAML.
     *
     * @throws MFEM_ABORT on any schema violation (missing root attrs,
     *         missing shot, missing receiver group, position size mismatch,
     *         unknown type string, …).
     */
    static ReceiverConfig::Config ReadReceivers(
        const std::string& path,
        int shot_id,
        const std::vector<std::string>& default_types,
        int expected_space_dim);

    /**
     * @brief Read source definitions from `/shots/<shot_id>/sources/`.
     *        Supports `force`, `pressure`, and `moment_tensor` types with
     *        scalar STF (rank-1 in time). Each `/stf` dataset is loaded
     *        as a (n_samples,) f64 array; rank-6 STF (`@layout="moment_tensor"`)
     *        is rejected — generalised MT inversion is a v2.1 concern.
     *
     * @throws MFEM_ABORT on schema violation (missing `@direction` for
     *         force, STF length mismatch, forbidden `@layout` / `@t0`,
     *         duplicate `@id`, missing `/moment_tensor` for MT types,
     *         invalid `@component_order`, unsupported `@coord_system`,
     *         etc.).
     */
    static HDF5SourceCatalog ReadSources(
        const std::string& path,
        int shot_id,
        int expected_space_dim);
};

}  // namespace SEM

#endif  // SEM_HDF5_SOURCE_RECEIVER_READER_HPP
