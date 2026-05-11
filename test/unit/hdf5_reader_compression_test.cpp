// hdf5_reader_compression_test — identical payload with/without gzip
// yields bit-identical ReadOwnedChannels output.

#include "srcrecv/HDF5ObservedReader.hpp"
#include "HDF5FixtureWriter.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace SEM;
using sem_test::MakeBasic2DSpec;
using sem_test::MakeTempH5Path;
using sem_test::PathGuard;
using sem_test::WriteFixture;

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        std::exit(1); \
    } \
} while (0)

int main() {
    const int N = 32;
    auto spec_u = MakeBasic2DSpec(N); spec_u.compress = false;
    auto spec_c = MakeBasic2DSpec(N); spec_c.compress = true;

    auto pu = MakeTempH5Path("cmp_u"); PathGuard gu{pu};
    auto pc = MakeTempH5Path("cmp_c"); PathGuard gc{pc};
    WriteFixture(pu, spec_u);
    WriteFixture(pc, spec_c);

    std::vector<HDF5ObservedReader::OwnedChannelRequest> reqs = {
        {"R0001", ReceiverType::Pressure,  -1, false},
        {"R0001", ReceiverType::Velocity,   0, false},
        {"R0001", ReceiverType::Velocity,   1, false},
        {"R0002", ReceiverType::Velocity,   0, false},
        {"R0003", ReceiverType::Pressure,  -1, true},
    };

    auto ru = HDF5ObservedReader::ReadOwnedChannels(pu, reqs);
    auto rc = HDF5ObservedReader::ReadOwnedChannels(pc, reqs);
    REQUIRE(ru.size() == rc.size());

    for (size_t i = 0; i < ru.size(); ++i) {
        REQUIRE(ru[i].data.size() == rc[i].data.size());
        for (size_t s = 0; s < ru[i].data.size(); ++s) {
            REQUIRE(ru[i].data[s] == rc[i].data[s]);
        }
        REQUIRE(ru[i].has_weight == rc[i].has_weight);
        REQUIRE(ru[i].weight.size() == rc[i].weight.size());
        for (size_t s = 0; s < ru[i].weight.size(); ++s) {
            REQUIRE(ru[i].weight[s] == rc[i].weight[s]);
        }
    }

    // Catalog identical too.
    auto cu = HDF5ObservedReader::ReadCatalog(pu, 2);
    auto cc = HDF5ObservedReader::ReadCatalog(pc, 2);
    REQUIRE(cu.n_samples == cc.n_samples);
    REQUIRE(cu.space_dim == cc.space_dim);
    REQUIRE(cu.dt == cc.dt);
    REQUIRE(cu.t0 == cc.t0);
    REQUIRE(cu.receivers.size() == cc.receivers.size());
    return 0;
}
