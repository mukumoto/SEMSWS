// hdf5_reader_catalog_test — ReadCatalog happy/error paths.

#include "srcrecv/HDF5ObservedReader.hpp"
#include "HDF5FixtureWriter.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

using namespace SEM;
using sem_test::FixtureSpec;
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

// Spawn this process in a subprocess and require it to fail with non-zero
// exit. We use fork() to detect MFEM_ABORT (which calls abort/exit).
#include <sys/wait.h>
#include <unistd.h>

template <typename F>
bool CallAborts(F&& fn) {
    pid_t pid = ::fork();
    if (pid == 0) {
        // Silence abort stderr.
        (void)!::freopen("/dev/null", "w", stderr);
        try { fn(); } catch (...) { std::_Exit(0); }
        std::_Exit(0);
    }
    int status = 0;
    ::waitpid(pid, &status, 0);
    return !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

void TestHappyPath() {
    auto path = MakeTempH5Path("cat_ok");
    PathGuard guard{path};
    auto spec = MakeBasic2DSpec(8);
    WriteFixture(path, spec);

    auto cat = HDF5ObservedReader::ReadCatalog(path, 2);
    REQUIRE(cat.format_version == "2.0");
    REQUIRE(std::abs(cat.dt - 1e-3) < 1e-15);
    REQUIRE(cat.t0 == 0.0);
    REQUIRE(cat.n_samples == 8);
    REQUIRE(cat.space_dim == 2);
    REQUIRE(cat.receivers.size() == 3);

    // Receivers come back in HDF5 group order (alphabetical by default).
    // Just check by name: all three present, positions right, channel sets.
    for (const auto& r : cat.receivers) {
        if (r.name == "R0001") {
            REQUIRE(r.position.Size() == 2);
            REQUIRE(r.position(0) == 10.0);
            REQUIRE(r.position(1) == 20.0);
            // pressure + velocity, no weights
            bool saw_p = false, saw_v = false;
            for (const auto& ch : r.channels) {
                if (ch.type == ReceiverType::Pressure) {
                    saw_p = true; REQUIRE(!ch.has_weight);
                } else if (ch.type == ReceiverType::Velocity) {
                    saw_v = true; REQUIRE(!ch.has_weight);
                }
            }
            REQUIRE(saw_p); REQUIRE(saw_v);
        } else if (r.name == "R0002") {
            REQUIRE(r.channels.size() == 1);
            REQUIRE(r.channels[0].type == ReceiverType::Velocity);
            REQUIRE(!r.channels[0].has_weight);
        } else if (r.name == "R0003") {
            REQUIRE(r.channels.size() == 1);
            REQUIRE(r.channels[0].type == ReceiverType::Pressure);
            REQUIRE(r.channels[0].has_weight);
        } else {
            REQUIRE(!"unexpected receiver name");
        }
    }
}

void TestErrorPaths() {
    // Missing root attrs
    {
        auto spec = MakeBasic2DSpec(); spec.skip_dt = true;
        auto p = MakeTempH5Path("cat_no_dt"); PathGuard g{p};
        WriteFixture(p, spec);
        REQUIRE(CallAborts([&]{ HDF5ObservedReader::ReadCatalog(p, 2); }));
    }
    {
        auto spec = MakeBasic2DSpec(); spec.skip_format_version = true;
        auto p = MakeTempH5Path("cat_no_ver"); PathGuard g{p};
        WriteFixture(p, spec);
        REQUIRE(CallAborts([&]{ HDF5ObservedReader::ReadCatalog(p, 2); }));
    }
    {
        auto spec = MakeBasic2DSpec(); spec.skip_n_samples = true;
        auto p = MakeTempH5Path("cat_no_ns"); PathGuard g{p};
        WriteFixture(p, spec);
        REQUIRE(CallAborts([&]{ HDF5ObservedReader::ReadCatalog(p, 2); }));
    }

    // Wrong major version (now: 1.x is rejected; users must migrate)
    {
        auto spec = MakeBasic2DSpec(); spec.override_format_version = "1.0";
        auto p = MakeTempH5Path("cat_bad_ver"); PathGuard g{p};
        WriteFixture(p, spec);
        REQUIRE(CallAborts([&]{ HDF5ObservedReader::ReadCatalog(p, 2); }));
    }

    // space_dim mismatch
    {
        auto spec = MakeBasic2DSpec();
        auto p = MakeTempH5Path("cat_bad_sd"); PathGuard g{p};
        WriteFixture(p, spec);
        REQUIRE(CallAborts([&]{ HDF5ObservedReader::ReadCatalog(p, 3); }));
    }

    // Position size mismatch
    {
        auto spec = MakeBasic2DSpec();
        spec.override_position_size = 3;  // but space_dim=2
        auto p = MakeTempH5Path("cat_bad_pos"); PathGuard g{p};
        WriteFixture(p, spec);
        REQUIRE(CallAborts([&]{ HDF5ObservedReader::ReadCatalog(p, 2); }));
    }

    // Vector dataset shape[0] != space_dim
    {
        auto spec = MakeBasic2DSpec();
        spec.override_vector_dim0 = 3;   // velocity[3, nt] in a 2D file
        auto p = MakeTempH5Path("cat_bad_vec"); PathGuard g{p};
        WriteFixture(p, spec);
        REQUIRE(CallAborts([&]{ HDF5ObservedReader::ReadCatalog(p, 2); }));
    }

    // weight_pressure with wrong length
    {
        auto spec = MakeBasic2DSpec();
        spec.override_weight_nsamples = spec.n_samples + 1;
        // Need to grow pressure_weight buffer to match bogus n_samples.
        for (auto& r : spec.receivers) {
            if (!r.pressure_weight.empty()) {
                r.pressure_weight.resize(spec.override_weight_nsamples, 0.0f);
            }
        }
        auto p = MakeTempH5Path("cat_bad_wp"); PathGuard g{p};
        WriteFixture(p, spec);
        REQUIRE(CallAborts([&]{ HDF5ObservedReader::ReadCatalog(p, 2); }));
    }

    // Receiver group with no channels
    {
        FixtureSpec spec;
        spec.space_dim = 2; spec.n_samples = 8;
        sem_test::ReceiverSpec r; r.name = "R0001"; r.position = {0.0, 0.0};
        // no datasets at all
        spec.receivers.push_back(r);
        auto p = MakeTempH5Path("cat_empty"); PathGuard g{p};
        WriteFixture(p, spec);
        REQUIRE(CallAborts([&]{ HDF5ObservedReader::ReadCatalog(p, 2); }));
    }

    // F14: units != "SI" → reject
    {
        auto spec = MakeBasic2DSpec();
        auto p = MakeTempH5Path("cat_units"); PathGuard g{p};
        WriteFixture(p, spec);
        // Re-open and overwrite the units attribute with "CGS".
        hid_t f = H5Fopen(p.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        if (H5Aexists(f, "units") > 0) {
            H5Adelete(f, "units");
        }
        sem_test::WriteStringAttr(f, "units", "CGS");
        H5Fclose(f);
        REQUIRE(CallAborts([&]{ HDF5ObservedReader::ReadCatalog(p, 2); }));
    }

    // F13: receiver group with an unknown dataset name → reject
    {
        auto spec = MakeBasic2DSpec();
        auto p = MakeTempH5Path("cat_grad"); PathGuard g{p};
        WriteFixture(p, spec);
        // Add a "gradient" dataset under R0001.
        hid_t f = H5Fopen(p.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        hid_t r0 = H5Gopen2(f, "/shots/0000/receivers/R0001", H5P_DEFAULT);
        hsize_t d[1] = { 8 };
        hid_t sp_id = H5Screate_simple(1, d, nullptr);
        hid_t ds = H5Dcreate2(r0, "gradient", H5T_NATIVE_FLOAT, sp_id,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dclose(ds); H5Sclose(sp_id);
        H5Gclose(r0); H5Fclose(f);
        REQUIRE(CallAborts([&]{ HDF5ObservedReader::ReadCatalog(p, 2); }));
    }
}

int main() {
    TestHappyPath();
    TestErrorPaths();
    return 0;
}
