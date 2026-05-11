// Receivers-only, geometry-only HDF5 input. Verifies that
// HDF5SourceReceiverReader::ReadReceivers turns a v2.0 fixture (positions
// + optional per-receiver `@types`) into a ReceiverConfig::Config matching
// what an inline YAML would produce.

#include "srcrecv/HDF5SourceReceiverReader.hpp"

#include <hdf5.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        std::exit(1); \
    } \
} while (0)

namespace {

void WriteScalarAttr(hid_t loc, const char* name, hid_t type, const void* v) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name, type, sp, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, type, v);
    H5Aclose(a);
    H5Sclose(sp);
}
void WriteStrAttr(hid_t loc, const char* name, const std::string& s) {
    hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, H5T_VARIABLE);
    H5Tset_cset(t, H5T_CSET_UTF8);
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name, t, sp, H5P_DEFAULT, H5P_DEFAULT);
    const char* cstr = s.c_str();
    H5Awrite(a, t, &cstr);
    H5Aclose(a); H5Sclose(sp); H5Tclose(t);
}
void WriteStrListAttr(hid_t loc, const char* name,
                      const std::vector<std::string>& vs) {
    hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, H5T_VARIABLE);
    H5Tset_cset(t, H5T_CSET_UTF8);
    hsize_t dims[1] = { static_cast<hsize_t>(vs.size()) };
    hid_t sp = H5Screate_simple(1, dims, nullptr);
    hid_t a = H5Acreate2(loc, name, t, sp, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<const char*> cstrs(vs.size());
    for (size_t i = 0; i < vs.size(); ++i) cstrs[i] = vs[i].c_str();
    H5Awrite(a, t, cstrs.data());
    H5Aclose(a); H5Sclose(sp); H5Tclose(t);
}
void WritePosAttr(hid_t loc, const std::vector<double>& pos) {
    hsize_t d[1] = { static_cast<hsize_t>(pos.size()) };
    hid_t sp = H5Screate_simple(1, d, nullptr);
    hid_t a = H5Acreate2(loc, "position", H5T_NATIVE_DOUBLE, sp,
                         H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, pos.data());
    H5Aclose(a); H5Sclose(sp);
}

std::string TempPath(const char* tag) {
    const char* tmp = std::getenv("TMPDIR");
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s/sem_srreader_%s_%d.h5",
                  tmp ? tmp : "/tmp", tag, static_cast<int>(::getpid()));
    return buf;
}

struct Cleanup {
    std::string path;
    ~Cleanup() { if (!path.empty()) std::remove(path.c_str()); }
};

// Build a minimal v2.0 file with three 2D receivers under /shots/0007/.
void WriteFixture(const std::string& path,
                  bool include_types_attr_for_R0002) {
    hid_t f = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    WriteStrAttr(f, "format_version", "2.0");
    double dt = 1e-3, t0 = 0.0;
    int64_t ns = 8;
    int sd = 2;
    WriteScalarAttr(f, "dt", H5T_NATIVE_DOUBLE, &dt);
    WriteScalarAttr(f, "t0", H5T_NATIVE_DOUBLE, &t0);
    WriteScalarAttr(f, "n_samples", H5T_NATIVE_INT64, &ns);
    WriteScalarAttr(f, "space_dim", H5T_NATIVE_INT32, &sd);

    hid_t g_shots = H5Gcreate2(f, "shots", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t g_shot  = H5Gcreate2(g_shots, "0007", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    int sid = 7;
    WriteScalarAttr(g_shot, "shot_id", H5T_NATIVE_INT32, &sid);
    hid_t g_recv = H5Gcreate2(g_shot, "receivers", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // R0001 — uses default types (no @types attr), with @label for friendly name.
    {
        hid_t g = H5Gcreate2(g_recv, "R0001", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        WritePosAttr(g, {10.0, 20.0});
        WriteStrAttr(g, "label", "STA_NORTH");
        H5Gclose(g);
    }
    // R0002 — overrides default types (when include_types_attr_for_R0002).
    {
        hid_t g = H5Gcreate2(g_recv, "R0002", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        WritePosAttr(g, {30.0, 40.0});
        if (include_types_attr_for_R0002) {
            WriteStrListAttr(g, "types", {"PS"});
        }
        H5Gclose(g);
    }
    // R0003 — no @label, no @types.
    {
        hid_t g = H5Gcreate2(g_recv, "R0003", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        WritePosAttr(g, {50.0, 60.0});
        H5Gclose(g);
    }

    H5Gclose(g_recv);
    H5Gclose(g_shot);
    H5Gclose(g_shots);
    H5Fclose(f);
}

template <typename F>
bool CallAborts(F&& fn) {
    pid_t pid = ::fork();
    if (pid == 0) {
        (void)!::freopen("/dev/null", "w", stderr);
        try { fn(); } catch (...) { std::_Exit(0); }
        std::_Exit(0);
    }
    int status = 0;
    ::waitpid(pid, &status, 0);
    return !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

}  // anonymous namespace

void TestHappyPath_DefaultTypes() {
    std::string path = TempPath("default_types");
    Cleanup g{path};
    WriteFixture(path, /*include_types_attr_for_R0002=*/false);

    auto cfg = SEM::HDF5SourceReceiverReader::ReadReceivers(
        path, /*shot_id=*/7, /*default_types=*/{"VEL"}, /*space_dim=*/2);

    REQUIRE(cfg.receivers.size() == 3);

    // HDF5 group order is alphabetical: R0001, R0002, R0003.
    REQUIRE(cfg.receivers[0].name == "STA_NORTH");           // @label override
    REQUIRE(cfg.receivers[0].location.size() == 2);
    REQUIRE(cfg.receivers[0].location[0] == 10.0);
    REQUIRE(cfg.receivers[0].location[1] == 20.0);
    REQUIRE(cfg.receivers[0].types == std::vector<std::string>{"VEL"});

    REQUIRE(cfg.receivers[1].name == "R0002");               // no label → group key
    REQUIRE(cfg.receivers[1].types == std::vector<std::string>{"VEL"});

    REQUIRE(cfg.receivers[2].name == "R0003");
    REQUIRE(cfg.receivers[2].types == std::vector<std::string>{"VEL"});
}

void TestPerReceiverTypeOverride() {
    std::string path = TempPath("type_override");
    Cleanup g{path};
    WriteFixture(path, /*include_types_attr_for_R0002=*/true);

    auto cfg = SEM::HDF5SourceReceiverReader::ReadReceivers(
        path, /*shot_id=*/7, {"VEL", "DISP"}, 2);

    REQUIRE(cfg.receivers.size() == 3);
    // R0002 has @types=["PS"] — overrides the parent-level default.
    REQUIRE(cfg.receivers[1].name == "R0002");
    REQUIRE(cfg.receivers[1].types == std::vector<std::string>{"PS"});
    // R0001 / R0003 still inherit defaults (already in YAML short form).
    REQUIRE(cfg.receivers[0].types ==
            (std::vector<std::string>{"VEL", "DISP"}));
    REQUIRE(cfg.receivers[2].types ==
            (std::vector<std::string>{"VEL", "DISP"}));
}

void TestErrorPaths() {
    // Missing shot.
    {
        std::string path = TempPath("err_no_shot");
        Cleanup g{path};
        WriteFixture(path, false);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadReceivers(
                path, /*shot_id=*/99, {"VEL"}, 2);
        }));
    }
    // Wrong space_dim.
    {
        std::string path = TempPath("err_bad_dim");
        Cleanup g{path};
        WriteFixture(path, false);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadReceivers(
                path, 7, {"VEL"}, /*space_dim=*/3);
        }));
    }
    // Unknown type string in default_types.
    {
        std::string path = TempPath("err_bad_type");
        Cleanup g{path};
        WriteFixture(path, false);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadReceivers(
                path, 7, {"bogus_type"}, 2);
        }));
    }
    // v1.0 file is rejected.
    {
        std::string path = TempPath("err_v1");
        Cleanup g{path};
        hid_t f = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        WriteStrAttr(f, "format_version", "1.0");
        double dt = 1e-3, t0 = 0.0; int64_t ns = 8; int sd = 2;
        WriteScalarAttr(f, "dt", H5T_NATIVE_DOUBLE, &dt);
        WriteScalarAttr(f, "t0", H5T_NATIVE_DOUBLE, &t0);
        WriteScalarAttr(f, "n_samples", H5T_NATIVE_INT64, &ns);
        WriteScalarAttr(f, "space_dim", H5T_NATIVE_INT32, &sd);
        H5Fclose(f);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadReceivers(
                path, 0, {"VEL"}, 2);
        }));
    }
}

int main() {
    TestHappyPath_DefaultTypes();
    TestPerReceiverTypeOverride();
    TestErrorPaths();
    return 0;
}
