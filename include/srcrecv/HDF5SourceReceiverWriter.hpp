/**
 * @file HDF5SourceReceiverWriter.hpp
 * @brief Writer-side helpers that emit `/shots/<NNNN>/sources/...` into a
 *        v2.0 SEMSWS HDF5 file. Receiver-side writing lives in
 *        ReceiverArray::SaveToHDF5; the helpers in this file plug into
 *        that path so a single per-shot file can carry both source
 *        metadata and receiver waveforms (self-roundtrip).
 */

#ifndef SEM_HDF5_SOURCE_RECEIVER_WRITER_HPP
#define SEM_HDF5_SOURCE_RECEIVER_WRITER_HPP

#include "common/Types.hpp"
#include "config/ConfigTypes.hpp"

#include <hdf5.h>
#include <string>
#include <vector>

namespace SEM {

/// Single-source descriptor used by the writer. Built from
/// `SourceConfig::Config{2D,3D}` plus the scalar STF samples
/// (`SourceTimeFunction::FromConfig(...)` output).
struct HDF5SourceWriteEntry {
    int id = 0;
    std::string label;                   ///< @label (optional, may be empty)
    std::string type;                    ///< "force" | "pressure" | "moment_tensor"
    std::vector<real_t> position;        ///< size == space_dim
    std::vector<real_t> direction;       ///< force only; size == space_dim
    /// Canonical order: 3D = {Mxx,Myy,Mzz,Mxy,Mxz,Myz}, 2D = {Mxx,Myy,Mxy}.
    /// Empty for non-MT types.
    std::vector<real_t> moment_tensor;
    std::vector<real_t> stf;             ///< scalar (n_samples,)
};

/// Build descriptors from a 2D/3D SourceConfig. Generates the scalar STF
/// samples by calling `SourceTimeFunction::FromConfig` on each wavelet
/// (which already supports `type="hdf5"` with pre-loaded `stf_samples`).
std::vector<HDF5SourceWriteEntry>
BuildSourceWriteEntries(const SourceConfig::Config2D& cfg,
                        int n_samples, real_t dt);

std::vector<HDF5SourceWriteEntry>
BuildSourceWriteEntries(const SourceConfig::Config3D& cfg,
                        int n_samples, real_t dt);

/// Emit `/shots/<shot_id>/sources/<S0001>/...` for each entry into the
/// already-open shot group `shot_group` (HDF5 group hid_t). Caller is
/// responsible for opening / closing the shot group itself.
///
/// Rank-0 only — the caller (ReceiverArray::SaveToHDF5) gates this.
void WriteSourcesIntoShotGroup(hid_t shot_group,
                               int space_dim,
                               const std::vector<HDF5SourceWriteEntry>& entries);

}  // namespace SEM

#endif  // SEM_HDF5_SOURCE_RECEIVER_WRITER_HPP
