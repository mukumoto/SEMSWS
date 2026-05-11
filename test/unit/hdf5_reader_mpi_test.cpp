// hdf5_reader_mpi_test — each rank opens the same HDF5 file with an
// independent read-only handle and reads a disjoint subset of channels via
// hyperslabs. Validates that Phase-3 of ObservedData works on serial HDF5
// across multiple MPI ranks on a shared filesystem (no Parallel HDF5).

#include "srcrecv/HDF5ObservedReader.hpp"
#include "HDF5FixtureWriter.hpp"

#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace SEM;
using sem_test::MakeBasic2DSpec;
using sem_test::WriteFixture;

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "[rank %d] FAIL %s:%d: %s\n", \
                     g_rank, __FILE__, __LINE__, #expr); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

static int g_rank = 0;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    g_rank = rank;

    // Rank 0 writes a shared fixture; all ranks read from the same path.
    const int N = 32;
    char path[512];
    std::snprintf(path, sizeof(path), "/tmp/semsws_mpi_fixture_%d.h5",
                  static_cast<int>(getpid()) ^ 0xdead);
    // Use rank 0's PID so every rank sees the same name.
    MPI_Bcast(path, sizeof(path), MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto spec = MakeBasic2DSpec(N);
        WriteFixture(path, spec);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Enumerate all (receiver, channel, component) triples in the fixture.
    // Order matches MakeBasic2DSpec:
    //   R0001/pressure  (-1)
    //   R0001/velocity[0], velocity[1]
    //   R0002/velocity[0], velocity[1]
    //   R0003/pressure  (-1)    has weight
    struct Triple {
        const char* name;
        ReceiverType type;
        int component;
        bool has_weight;
    };
    const Triple all[] = {
        {"R0001", ReceiverType::Pressure,  -1, false},
        {"R0001", ReceiverType::Velocity,   0, false},
        {"R0001", ReceiverType::Velocity,   1, false},
        {"R0002", ReceiverType::Velocity,   0, false},
        {"R0002", ReceiverType::Velocity,   1, false},
        {"R0003", ReceiverType::Pressure,  -1, true },
    };
    const int ntriples = sizeof(all) / sizeof(all[0]);

    // Round-robin assign triples to ranks.
    std::vector<HDF5ObservedReader::OwnedChannelRequest> my_reqs;
    std::vector<int> my_triple_idx;
    for (int i = 0; i < ntriples; ++i) {
        if (i % nprocs != rank) continue;
        HDF5ObservedReader::OwnedChannelRequest req;
        req.receiver_name = all[i].name;
        req.type          = all[i].type;
        req.component     = all[i].component;
        req.want_weight   = all[i].has_weight;
        my_reqs.push_back(req);
        my_triple_idx.push_back(i);
    }

    // Each rank opens the file independently.
    auto results = HDF5ObservedReader::ReadOwnedChannels(path, my_reqs);
    REQUIRE(results.size() == my_reqs.size());

    // Check per-rank correctness against the known generator formulas.
    for (size_t k = 0; k < my_reqs.size(); ++k) {
        const auto& req = my_reqs[k];
        const auto& res = results[k];
        REQUIRE(static_cast<int>(res.data.size()) == N);

        if (req.receiver_name == "R0001" && req.type == ReceiverType::Pressure) {
            for (int i = 0; i < N; ++i) {
                REQUIRE(res.data[i] == 1.0f + 0.125f * i);
            }
        } else if (req.receiver_name == "R0001"
                   && req.type == ReceiverType::Velocity) {
            // v[c,i] = 100*(c+1) + i
            float base = 100.0f * (req.component + 1);
            for (int i = 0; i < N; ++i) REQUIRE(res.data[i] == base + i);
        } else if (req.receiver_name == "R0002"
                   && req.type == ReceiverType::Velocity) {
            // v[c,i] = -5*(c+1) + 0.5*i
            float base = -5.0f * (req.component + 1);
            for (int i = 0; i < N; ++i) {
                REQUIRE(res.data[i] == base + 0.5f * i);
            }
        } else if (req.receiver_name == "R0003"
                   && req.type == ReceiverType::Pressure) {
            REQUIRE(res.has_weight);
            for (int i = 0; i < N; ++i) {
                REQUIRE(res.data[i]   == 7.0f  + 0.125f * i);
                REQUIRE(res.weight[i] == 0.25f + 0.1f   * i);
            }
        } else {
            REQUIRE(!"unexpected (receiver,type) combination");
        }
    }

    // Cross-rank sanity: gather (triple_idx, data[0]) to rank 0, make sure
    // every triple was covered exactly once, matching reference values.
    int my_count = static_cast<int>(my_triple_idx.size());
    std::vector<int> counts(nprocs, 0);
    MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0,
               MPI_COMM_WORLD);
    std::vector<int> displs(nprocs, 0);
    int total = 0;
    if (rank == 0) {
        for (int r = 0; r < nprocs; ++r) {
            displs[r] = total;
            total += counts[r];
        }
        REQUIRE(total == ntriples);
    }
    std::vector<int> all_idx(total, -1);
    MPI_Gatherv(my_triple_idx.data(), my_count, MPI_INT,
                all_idx.data(), counts.data(), displs.data(), MPI_INT, 0,
                MPI_COMM_WORLD);
    if (rank == 0) {
        std::vector<int> seen(ntriples, 0);
        for (int v : all_idx) {
            REQUIRE(v >= 0 && v < ntriples);
            seen[v]++;
        }
        for (int i = 0; i < ntriples; ++i) REQUIRE(seen[i] == 1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::remove(path);

    MPI_Finalize();
    return 0;
}
