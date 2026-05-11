// hdf5_srcrecv_reader_sources_test — sources-only reader path:
// force / pressure types with scalar STF, plus moment-tensor variants.

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
void WriteVecAttr(hid_t loc, const char* name,
                  const std::vector<double>& vs) {
    hsize_t d[1] = { static_cast<hsize_t>(vs.size()) };
    hid_t sp = H5Screate_simple(1, d, nullptr);
    hid_t a = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, sp,
                         H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, H5T_NATIVE_DOUBLE, vs.data());
    H5Aclose(a); H5Sclose(sp);
}
void WriteStfDset(hid_t loc, const std::vector<double>& samples) {
    hsize_t d[1] = { static_cast<hsize_t>(samples.size()) };
    hid_t sp = H5Screate_simple(1, d, nullptr);
    hid_t ds = H5Dcreate2(loc, "stf", H5T_NATIVE_DOUBLE, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             samples.data());
    H5Dclose(ds);
    H5Sclose(sp);
}

std::string TempPath(const char* tag) {
    const char* tmp = std::getenv("TMPDIR");
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s/sem_srcread_%s_%d.h5",
                  tmp ? tmp : "/tmp", tag, static_cast<int>(::getpid()));
    return buf;
}

struct Cleanup {
    std::string path;
    ~Cleanup() { if (!path.empty()) std::remove(path.c_str()); }
};

struct FixtureKnobs {
    bool force_skip_direction = false;
    bool stf_wrong_length = false;
    bool stf_has_layout_attr = false;
    bool source_has_t0_attr = false;
    bool make_mt_source = false;       // pressure → moment_tensor (happy path)
    bool duplicate_id = false;
    bool wrong_group_key = false;      // S5 instead of S0001
    bool empty_sources = false;
    int n_samples = 16;
    int space_dim = 2;
    int shot_id = 3;
    int force_id = 1;
    int pressure_id = 2;
    // MT-specific knobs (only meaningful when make_mt_source).
    bool mt_skip_dataset = false;
    bool mt_wrong_shape = false;       // shape (4,) instead of (3,) or (6,)
    bool mt_skip_component_order = false;
    bool mt_invalid_component = false; // component_order has "Mab"
    bool mt_duplicate_component = false;
    bool mt_unsupported_coord = false; // @coord_system="rtp"
    std::vector<std::string> mt_component_order_override;  // empty → canonical
};

// Build a minimal v2.0 file with two 2D sources (force + pressure) under
// /shots/0003/sources/. Knobs flip individual fields for error tests.
void WriteFixture(const std::string& path, const FixtureKnobs& kn = {}) {
    hid_t f = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    WriteStrAttr(f, "format_version", "2.0");
    double dt = 1e-3, t0 = 0.0;
    int64_t ns = kn.n_samples;
    int sd = kn.space_dim;
    WriteScalarAttr(f, "dt", H5T_NATIVE_DOUBLE, &dt);
    WriteScalarAttr(f, "t0", H5T_NATIVE_DOUBLE, &t0);
    WriteScalarAttr(f, "n_samples", H5T_NATIVE_INT64, &ns);
    WriteScalarAttr(f, "space_dim", H5T_NATIVE_INT32, &sd);

    hid_t g_shots = H5Gcreate2(f, "shots", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    char shot_key[8];
    std::snprintf(shot_key, sizeof(shot_key), "%04d", kn.shot_id);
    hid_t g_shot = H5Gcreate2(g_shots, shot_key, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    int sid = kn.shot_id;
    WriteScalarAttr(g_shot, "shot_id", H5T_NATIVE_INT32, &sid);
    hid_t g_src = H5Gcreate2(g_shot, "sources", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    auto pad_to_dim = [&](std::vector<double> v) {
        v.resize(static_cast<size_t>(kn.space_dim), 0.0);
        return v;
    };

    auto write_force = [&](int id, const char* groupkey,
                           const std::vector<double>& pos_in) {
        hid_t g = H5Gcreate2(g_src, groupkey, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        int id_v = id;
        WriteScalarAttr(g, "id", H5T_NATIVE_INT32, &id_v);
        WriteStrAttr(g, "label", "force_test");
        WriteStrAttr(g, "type", "force");
        WriteVecAttr(g, "position", pad_to_dim(pos_in));
        if (!kn.force_skip_direction) {
            std::vector<double> dir(kn.space_dim, 0.0);
            dir[kn.space_dim - 1] = -1.0;
            WriteVecAttr(g, "direction", dir);
        }
        if (kn.source_has_t0_attr) {
            double per_t0 = 0.5;
            WriteScalarAttr(g, "t0", H5T_NATIVE_DOUBLE, &per_t0);
        }
        // STF: linear ramp. Use 0.5*i to keep values exactly representable
        // in both single and double precision.
        const int n = kn.stf_wrong_length ? (kn.n_samples + 3) : kn.n_samples;
        std::vector<double> samples(n);
        for (int i = 0; i < n; ++i) {
            samples[i] = 0.5 * static_cast<double>(i);
        }
        // Build dset manually so we can stick @layout on it.
        hsize_t d[1] = { static_cast<hsize_t>(n) };
        hid_t sp = H5Screate_simple(1, d, nullptr);
        hid_t ds = H5Dcreate2(g, "stf", H5T_NATIVE_DOUBLE, sp,
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                 samples.data());
        if (kn.stf_has_layout_attr) {
            WriteStrAttr(ds, "layout", "moment_tensor");
        }
        H5Dclose(ds);
        H5Sclose(sp);
        H5Gclose(g);
    };

    auto write_pressure = [&](int id, const char* groupkey,
                              const std::vector<double>& pos_in) {
        hid_t g = H5Gcreate2(g_src, groupkey, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        int id_v = id;
        WriteScalarAttr(g, "id", H5T_NATIVE_INT32, &id_v);
        WriteStrAttr(g, "type", "pressure");
        WriteVecAttr(g, "position", pad_to_dim(pos_in));
        std::vector<double> samples(kn.n_samples);
        for (int i = 0; i < kn.n_samples; ++i) samples[i] = -0.5 * i;
        WriteStfDset(g, samples);
        H5Gclose(g);
    };

    auto write_str_list = [](hid_t loc, const char* name,
                             const std::vector<std::string>& vs) {
        hid_t t = H5Tcopy(H5T_C_S1);
        H5Tset_size(t, H5T_VARIABLE);
        H5Tset_cset(t, H5T_CSET_UTF8);
        hsize_t d[1] = { static_cast<hsize_t>(vs.size()) };
        hid_t sp = H5Screate_simple(1, d, nullptr);
        hid_t a = H5Acreate2(loc, name, t, sp, H5P_DEFAULT, H5P_DEFAULT);
        std::vector<const char*> cstrs(vs.size());
        for (size_t i = 0; i < vs.size(); ++i) cstrs[i] = vs[i].c_str();
        H5Awrite(a, t, cstrs.data());
        H5Aclose(a); H5Sclose(sp); H5Tclose(t);
    };

    auto write_mt = [&](int id, const char* groupkey,
                        const std::vector<double>& pos_in,
                        const std::vector<double>& mt_values_in_file_order,
                        const std::vector<std::string>& component_order) {
        hid_t g = H5Gcreate2(g_src, groupkey, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        int id_v = id;
        WriteScalarAttr(g, "id", H5T_NATIVE_INT32, &id_v);
        WriteStrAttr(g, "label", "mt_test");
        WriteStrAttr(g, "type", "moment_tensor");
        WriteVecAttr(g, "position", pad_to_dim(pos_in));

        if (!kn.mt_skip_dataset) {
            const int n_eff = kn.mt_wrong_shape
                ? 4 : static_cast<int>(mt_values_in_file_order.size());
            hsize_t d[1] = { static_cast<hsize_t>(n_eff) };
            hid_t sp = H5Screate_simple(1, d, nullptr);
            hid_t ds = H5Dcreate2(g, "moment_tensor", H5T_NATIVE_DOUBLE, sp,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            std::vector<double> buf(n_eff, 0.0);
            for (size_t i = 0;
                 i < mt_values_in_file_order.size() && static_cast<int>(i) < n_eff;
                 ++i) {
                buf[i] = mt_values_in_file_order[i];
            }
            H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                     buf.data());
            if (!kn.mt_skip_component_order) {
                write_str_list(ds, "component_order", component_order);
            }
            if (kn.mt_unsupported_coord) {
                WriteStrAttr(ds, "coord_system", "rtp");
            }
            H5Dclose(ds);
            H5Sclose(sp);
        }

        std::vector<double> samples(kn.n_samples);
        for (int i = 0; i < kn.n_samples; ++i) samples[i] = 0.25 * i;
        WriteStfDset(g, samples);
        H5Gclose(g);
    };

    if (!kn.empty_sources) {
        const char* fkey = kn.wrong_group_key ? "S5" : "S0001";
        write_force(kn.force_id, fkey, {100.0, 200.0});
        if (kn.duplicate_id) {
            // second force with same id ⇒ F9 violation
            write_force(kn.force_id, "S0099", {0.0, 0.0});
        } else if (kn.make_mt_source) {
            // 2D MT canonical {Mxx, Myy, Mxy} → values {1.0, 2.0, 3.0}.
            // 3D MT canonical {Mxx, Myy, Mzz, Mxy, Mxz, Myz} → {1..6}.
            std::vector<double> canon_vals = (kn.space_dim == 3)
                ? std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
                : std::vector<double>{1.0, 2.0, 3.0};
            std::vector<std::string> canon_order = (kn.space_dim == 3)
                ? std::vector<std::string>{"Mxx","Myy","Mzz","Mxy","Mxz","Myz"}
                : std::vector<std::string>{"Mxx","Myy","Mxy"};

            std::vector<double> file_vals = canon_vals;
            std::vector<std::string> file_order = canon_order;

            // Override component_order (and shuffle values to match) when
            // requested by the caller; otherwise emit canonical layout.
            if (!kn.mt_component_order_override.empty()) {
                file_order = kn.mt_component_order_override;
                // For each entry in file_order, lookup its canonical value.
                file_vals.clear();
                for (const auto& nm : file_order) {
                    bool ok = false;
                    for (size_t k = 0; k < canon_order.size(); ++k) {
                        if (canon_order[k] == nm) {
                            file_vals.push_back(canon_vals[k]);
                            ok = true;
                            break;
                        }
                    }
                    if (!ok) file_vals.push_back(0.0);
                }
            }
            if (kn.mt_invalid_component && !file_order.empty()) {
                file_order[0] = "Mab";
            }
            if (kn.mt_duplicate_component && file_order.size() >= 2) {
                file_order[1] = file_order[0];
            }

            write_mt(kn.pressure_id, "S0002", {0.0, 0.0},
                     file_vals, file_order);
        } else {
            write_pressure(kn.pressure_id, "S0002", {300.0, 400.0});
        }
    }

    H5Gclose(g_src);
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

void TestHappyPath() {
    std::string path = TempPath("happy");
    Cleanup g{path};
    WriteFixture(path);

    auto cat = SEM::HDF5SourceReceiverReader::ReadSources(path, /*shot=*/3,
                                                          /*space_dim=*/2);
    REQUIRE(cat.n_samples == 16);
    REQUIRE(cat.space_dim == 2);
    REQUIRE(cat.sources.size() == 2);

    // S0001 (force)
    REQUIRE(cat.sources[0].id == 1);
    REQUIRE(cat.sources[0].label == "force_test");
    REQUIRE(cat.sources[0].type == "force");
    REQUIRE(cat.sources[0].position.size() == 2);
    REQUIRE(cat.sources[0].position[0] == 100.0);
    REQUIRE(cat.sources[0].position[1] == 200.0);
    REQUIRE(cat.sources[0].direction.size() == 2);
    REQUIRE(cat.sources[0].direction[1] == -1.0);
    REQUIRE(static_cast<int>(cat.sources[0].stf.size()) == 16);
    REQUIRE(cat.sources[0].stf[3] == 1.5);  // 0.5 * 3, exact in float/double

    // S0002 (pressure) — no @label attr → label empty
    REQUIRE(cat.sources[1].id == 2);
    REQUIRE(cat.sources[1].label.empty());
    REQUIRE(cat.sources[1].type == "pressure");
    REQUIRE(cat.sources[1].position[0] == 300.0);
    REQUIRE(cat.sources[1].stf[2] == -1.0);  // -0.5 * 2, exact
}

void TestErrorPaths() {
    // Force without @direction
    {
        std::string path = TempPath("nodir");
        Cleanup c{path};
        FixtureKnobs k; k.force_skip_direction = true;
        WriteFixture(path, k);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 2);
        }));
    }
    // STF length mismatch
    {
        std::string path = TempPath("badlen");
        Cleanup c{path};
        FixtureKnobs k; k.stf_wrong_length = true;
        WriteFixture(path, k);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 2);
        }));
    }
    // F2: /stf with @layout attribute
    {
        std::string path = TempPath("layout");
        Cleanup c{path};
        FixtureKnobs k; k.stf_has_layout_attr = true;
        WriteFixture(path, k);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 2);
        }));
    }
    // F7: per-source @t0
    {
        std::string path = TempPath("pst0");
        Cleanup c{path};
        FixtureKnobs k; k.source_has_t0_attr = true;
        WriteFixture(path, k);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 2);
        }));
    }
    // MT-specific error paths are covered by TestMomentTensorErrorPaths
    // below.
    // F9: duplicate @id
    {
        std::string path = TempPath("dupid");
        Cleanup c{path};
        FixtureKnobs k; k.duplicate_id = true;
        WriteFixture(path, k);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 2);
        }));
    }
    // F8: group key mismatch (S5 vs S0001)
    {
        std::string path = TempPath("badkey");
        Cleanup c{path};
        FixtureKnobs k; k.wrong_group_key = true;
        WriteFixture(path, k);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 2);
        }));
    }
    // F11: empty shot
    {
        std::string path = TempPath("empty");
        Cleanup c{path};
        FixtureKnobs k; k.empty_sources = true;
        WriteFixture(path, k);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 2);
        }));
    }
    // shot_id missing
    {
        std::string path = TempPath("noshot");
        Cleanup c{path};
        WriteFixture(path);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 99, 2);
        }));
    }
    // wrong space_dim
    {
        std::string path = TempPath("baddim");
        Cleanup c{path};
        WriteFixture(path);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 3);
        }));
    }
}

void TestMomentTensorHappyPath2D() {
    std::string path = TempPath("mt2d");
    Cleanup c{path};
    FixtureKnobs k; k.make_mt_source = true; k.space_dim = 2;
    WriteFixture(path, k);

    auto cat = SEM::HDF5SourceReceiverReader::ReadSources(path, /*shot=*/3,
                                                          /*space_dim=*/2);
    REQUIRE(cat.sources.size() == 2);
    const auto& mt = cat.sources[1];
    REQUIRE(mt.type == "moment_tensor");
    REQUIRE(mt.label == "mt_test");
    REQUIRE(mt.moment_tensor.size() == 3);
    // canonical 2D order: {Mxx, Myy, Mxy} = {1.0, 2.0, 3.0}
    REQUIRE(mt.moment_tensor[0] == 1.0);
    REQUIRE(mt.moment_tensor[1] == 2.0);
    REQUIRE(mt.moment_tensor[2] == 3.0);
    // STF should still be the linear ramp written by write_mt.
    REQUIRE(mt.stf.size() == 16);
    REQUIRE(mt.stf[4] == 1.0);  // 0.25 * 4
}

void TestMomentTensorHappyPath3D() {
    std::string path = TempPath("mt3d");
    Cleanup c{path};
    FixtureKnobs k; k.make_mt_source = true; k.space_dim = 3;
    WriteFixture(path, k);

    auto cat = SEM::HDF5SourceReceiverReader::ReadSources(path, /*shot=*/3,
                                                          /*space_dim=*/3);
    REQUIRE(cat.sources.size() == 2);
    const auto& mt = cat.sources[1];
    REQUIRE(mt.type == "moment_tensor");
    REQUIRE(mt.moment_tensor.size() == 6);
    // canonical 3D order: {Mxx, Myy, Mzz, Mxy, Mxz, Myz} = {1,2,3,4,5,6}
    for (int i = 0; i < 6; ++i) {
        REQUIRE(mt.moment_tensor[i] == static_cast<double>(i + 1));
    }
}

void TestMomentTensorComponentOrderShuffle() {
    // File writes components in {Mzz, Mxx, Myz, Mxy, Myy, Mxz} order with
    // values that match the canonical {1, 2, ..., 6} mapping. Reader must
    // remap to canonical so the output values are still {1..6}.
    std::string path = TempPath("mt_shuffle");
    Cleanup c{path};
    FixtureKnobs k;
    k.make_mt_source = true;
    k.space_dim = 3;
    k.mt_component_order_override =
        {"Mzz", "Mxx", "Myz", "Mxy", "Myy", "Mxz"};
    WriteFixture(path, k);

    auto cat = SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 3);
    REQUIRE(cat.sources.size() == 2);
    const auto& mt = cat.sources[1];
    // Canonical out should be {Mxx=1, Myy=2, Mzz=3, Mxy=4, Mxz=5, Myz=6}
    // — exactly the canonical values regardless of file order.
    for (int i = 0; i < 6; ++i) {
        REQUIRE(mt.moment_tensor[i] == static_cast<double>(i + 1));
    }
}

void TestMomentTensorErrorPaths() {
    // Missing /moment_tensor dataset.
    {
        std::string path = TempPath("mt_no_ds");
        Cleanup c{path};
        FixtureKnobs k; k.make_mt_source = true;
        k.space_dim = 2; k.mt_skip_dataset = true;
        WriteFixture(path, k);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 2);
        }));
    }
    // Wrong shape (4 components in 2D).
    {
        std::string path = TempPath("mt_bad_shape");
        Cleanup c{path};
        FixtureKnobs k; k.make_mt_source = true;
        k.space_dim = 2; k.mt_wrong_shape = true;
        WriteFixture(path, k);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 2);
        }));
    }
    // Missing @component_order.
    {
        std::string path = TempPath("mt_no_order");
        Cleanup c{path};
        FixtureKnobs k; k.make_mt_source = true;
        k.space_dim = 2; k.mt_skip_component_order = true;
        WriteFixture(path, k);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 2);
        }));
    }
    // Invalid component name in @component_order.
    {
        std::string path = TempPath("mt_bad_name");
        Cleanup c{path};
        FixtureKnobs k; k.make_mt_source = true;
        k.space_dim = 2; k.mt_invalid_component = true;
        WriteFixture(path, k);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 2);
        }));
    }
    // Duplicate component in @component_order.
    {
        std::string path = TempPath("mt_dup");
        Cleanup c{path};
        FixtureKnobs k; k.make_mt_source = true;
        k.space_dim = 2; k.mt_duplicate_component = true;
        WriteFixture(path, k);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 2);
        }));
    }
    // Unsupported @coord_system="rtp".
    {
        std::string path = TempPath("mt_coord");
        Cleanup c{path};
        FixtureKnobs k; k.make_mt_source = true;
        k.space_dim = 2; k.mt_unsupported_coord = true;
        WriteFixture(path, k);
        REQUIRE(CallAborts([&]{
            SEM::HDF5SourceReceiverReader::ReadSources(path, 3, 2);
        }));
    }
}

int main() {
    TestHappyPath();
    TestErrorPaths();
    TestMomentTensorHappyPath2D();
    TestMomentTensorHappyPath3D();
    TestMomentTensorComponentOrderShuffle();
    TestMomentTensorErrorPaths();
    return 0;
}
