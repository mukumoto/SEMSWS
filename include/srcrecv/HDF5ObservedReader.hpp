/**
 * @file HDF5ObservedReader.hpp
 * @brief Canonical HDF5 observed-data reader (rank-aware, no bulk broadcast).
 *
 * Schema (format_version 2.0; see include/srcrecv/HDF5IOSchema.hpp):
 *   root attrs: format_version, dt, t0, n_samples, space_dim, ...
 *   /shots/<NNNN>/receivers/<name>/
 *       attrs: position [space_dim]
 *       datasets (at least one required for observed input):
 *           velocity      [space_dim, n_samples]   optional
 *           displacement  [space_dim, n_samples]   optional
 *           acceleration  [space_dim, n_samples]   optional
 *           pressure      [n_samples]              optional
 *           weight_<ch>   same shape as channel    optional
 *
 * 1.x files are rejected; users must run the migration tool
 * `driver/tools/migrate_observed_v1_to_v2.py` to upgrade.
 *
 * Usage (two-phase):
 *   1. ReadCatalog(path, space_dim, shot_id=0) on rank 0 to discover
 *      structure and broadcast. No bulk reads.
 *   2. ReadOwnedChannels(path, requests, shot_id=0) per-rank to fetch only
 *      owned receiver/channel slabs via HDF5 hyperslabs.
 */

#ifndef SEM_HDF5_OBSERVED_READER_HPP
#define SEM_HDF5_OBSERVED_READER_HPP

#include <mfem.hpp>
#include <string>
#include <vector>
#include "common/Types.hpp"

namespace SEM {

using mfem::real_t;
using mfem::Vector;

struct HDF5ChannelDescriptor {
    ReceiverType type;     // pressure | velocity | displacement | acceleration
    bool has_weight;
};

struct HDF5ReceiverEntry {
    std::string name;                         // e.g. "R0001"
    Vector position;                          // size == space_dim
    std::vector<HDF5ChannelDescriptor> channels;
};

struct HDF5ObservedCatalog {
    std::string format_version;
    double dt = 0.0;
    double t0 = 0.0;
    int n_samples = 0;
    int space_dim = 0;
    std::vector<HDF5ReceiverEntry> receivers;
};

class HDF5ObservedReader {
public:
    /**
     * @brief Phase-1 metadata walk. Rank 0 only. No bulk data reads.
     *
     * @param shot_id Index into `/shots/<NNNN>/` (default 0). Files written
     *   by the per-shot writer always store data under `/shots/0000/`; the
     *   driver merge tool reorganises multiple per-shot files under
     *   `/shots/0000`, `/shots/0001`, ...
     * @throws MFEM_ABORT on schema violation.
     */
    static HDF5ObservedCatalog ReadCatalog(const std::string& path,
                                           int expected_space_dim,
                                           int shot_id = 0);

    /// Phase-3 hyperslab request descriptor.
    struct OwnedChannelRequest {
        std::string receiver_name;
        ReceiverType type;
        int component;        // -1 scalar; 0..space_dim-1 vector component
        bool want_weight;
    };

    /// Phase-3 result (parallel to requests vector).
    struct OwnedChannelResult {
        std::vector<float> data;
        std::vector<float> weight;
        bool has_weight = false;
    };

    /**
     * @brief Phase-3 per-rank hyperslab read.
     *
     * Opens `path` read-only with an independent handle, reads only the
     * requested (receiver, channel, component) 1-D slabs, closes the file.
     * Works with serial HDF5 on a shared filesystem (no Parallel HDF5).
     *
     * Returned buffers have length == root `n_samples`.
     *
     * @param shot_id Same meaning as in ReadCatalog (default 0).
     */
    static std::vector<OwnedChannelResult>
    ReadOwnedChannels(const std::string& path,
                      const std::vector<OwnedChannelRequest>& requests,
                      int shot_id = 0);
};

}  // namespace SEM

#endif  // SEM_HDF5_OBSERVED_READER_HPP
