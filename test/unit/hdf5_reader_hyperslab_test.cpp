// hdf5_reader_hyperslab_test — ReadOwnedChannels verify exact bytes.

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
    const int N = 16;
    auto spec = MakeBasic2DSpec(N);
    auto path = MakeTempH5Path("slab"); PathGuard guard{path};
    WriteFixture(path, spec);

    // Empty requests -> empty results (no crash).
    {
        auto r = HDF5ObservedReader::ReadOwnedChannels(path, {});
        REQUIRE(r.empty());
    }

    // Pressure scalar from R0001.
    {
        HDF5ObservedReader::OwnedChannelRequest req;
        req.receiver_name = "R0001";
        req.type = ReceiverType::Pressure;
        req.component = -1;
        req.want_weight = false;
        auto r = HDF5ObservedReader::ReadOwnedChannels(path, {req});
        REQUIRE(r.size() == 1);
        REQUIRE(static_cast<int>(r[0].data.size()) == N);
        REQUIRE(!r[0].has_weight);
        // Expected: R0001.pressure = 1.0 + 0.125*i.
        for (int i = 0; i < N; ++i) {
            float expect = 1.0f + 0.125f * i;
            REQUIRE(r[0].data[i] == expect);
        }
    }

    // Velocity component 1 from R0002.
    {
        HDF5ObservedReader::OwnedChannelRequest req;
        req.receiver_name = "R0002";
        req.type = ReceiverType::Velocity;
        req.component = 1;
        req.want_weight = false;
        auto r = HDF5ObservedReader::ReadOwnedChannels(path, {req});
        REQUIRE(r.size() == 1);
        // R0002.velocity[c,i] = -5*(c+1) + 0.5*i, so c=1 -> -10 + 0.5*i
        for (int i = 0; i < N; ++i) {
            float expect = -10.0f + 0.5f * i;
            REQUIRE(r[0].data[i] == expect);
        }
    }

    // Pressure with weight from R0003.
    {
        HDF5ObservedReader::OwnedChannelRequest req;
        req.receiver_name = "R0003";
        req.type = ReceiverType::Pressure;
        req.component = -1;
        req.want_weight = true;
        auto r = HDF5ObservedReader::ReadOwnedChannels(path, {req});
        REQUIRE(r.size() == 1);
        REQUIRE(r[0].has_weight);
        REQUIRE(static_cast<int>(r[0].weight.size()) == N);
        for (int i = 0; i < N; ++i) {
            REQUIRE(r[0].data[i] == 7.0f + 0.125f * i);
            REQUIRE(r[0].weight[i] == 0.25f + 0.1f * i);
        }
    }

    // Mixed: multiple requests in one call, verify order preserved.
    {
        std::vector<HDF5ObservedReader::OwnedChannelRequest> reqs = {
            {"R0001", ReceiverType::Velocity,  0, false},
            {"R0003", ReceiverType::Pressure, -1, true},
            {"R0001", ReceiverType::Velocity,  1, false},
        };
        auto r = HDF5ObservedReader::ReadOwnedChannels(path, reqs);
        REQUIRE(r.size() == 3);
        // R0001.velocity[0,i] = 100 + i
        for (int i = 0; i < N; ++i) REQUIRE(r[0].data[i] == 100.0f + i);
        // R0003.pressure
        for (int i = 0; i < N; ++i) REQUIRE(r[1].data[i] == 7.0f + 0.125f * i);
        REQUIRE(r[1].has_weight);
        // R0001.velocity[1,i] = 200 + i
        for (int i = 0; i < N; ++i) REQUIRE(r[2].data[i] == 200.0f + i);
    }

    // want_weight=true but no weight_<channel> dataset -> has_weight=false.
    {
        HDF5ObservedReader::OwnedChannelRequest req;
        req.receiver_name = "R0001";
        req.type = ReceiverType::Pressure;
        req.component = -1;
        req.want_weight = true;
        auto r = HDF5ObservedReader::ReadOwnedChannels(path, {req});
        REQUIRE(r.size() == 1);
        REQUIRE(!r[0].has_weight);
    }

    return 0;
}
