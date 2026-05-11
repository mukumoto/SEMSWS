// HDF5FixtureWriter.hpp — header-only helper for unit tests.
// Writes a v2.0 SEMSWS observed/source-receiver HDF5 file using the HDF5 C
// API (so headers match the actual 1.14 runtime; see note in
// HDF5ObservedReader.cpp).
//
// Layout: receivers go under `/shots/<NNNN>/receivers/<name>/...` per
// `include/srcrecv/HDF5IOSchema.hpp`. By default `shot_id=0`.

#ifndef SEM_TEST_HDF5_FIXTURE_WRITER_HPP
#define SEM_TEST_HDF5_FIXTURE_WRITER_HPP

#include <hdf5.h>
#include <string>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

namespace sem_test {

struct ReceiverSpec {
    std::string name;
    std::vector<double> position;

    std::vector<float> pressure;
    std::vector<float> pressure_weight;
    std::vector<float> velocity;              // dim0 * n_samples row-major
    std::vector<float> velocity_weight;
    std::vector<float> displacement;
    std::vector<float> displacement_weight;
    std::vector<float> acceleration;
    std::vector<float> acceleration_weight;
};

struct FixtureSpec {
    std::string format_version = "2.0";
    double dt = 1e-3;
    double t0 = 0.0;
    int n_samples = 8;
    int space_dim = 2;
    int shot_id = 0;
    std::vector<ReceiverSpec> receivers;
    bool compress = false;

    bool skip_format_version = false;
    bool skip_dt = false;
    bool skip_t0 = false;
    bool skip_n_samples = false;
    bool skip_space_dim = false;
    std::string override_format_version;
    int override_position_size = 0;
    int override_vector_dim0 = 0;
    int override_weight_nsamples = 0;
};

inline std::string MakeTempH5Path(const std::string& tag) {
    const char* tmp = std::getenv("TMPDIR");
    std::string base = tmp ? tmp : "/tmp";
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s/semsws_fixture_%s_%d.h5",
                  base.c_str(), tag.c_str(),
                  static_cast<int>(::getpid()));
    return buf;
}

inline void WriteScalarAttr(hid_t loc, const char* name, hid_t type,
                            const void* v) {
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name, type, sp, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(a, type, v);
    H5Aclose(a);
    H5Sclose(sp);
}

inline void WriteStringAttr(hid_t loc, const char* name, const std::string& s) {
    hid_t t = H5Tcopy(H5T_C_S1);
    H5Tset_size(t, H5T_VARIABLE);
    H5Tset_cset(t, H5T_CSET_UTF8);
    hid_t sp = H5Screate(H5S_SCALAR);
    hid_t a = H5Acreate2(loc, name, t, sp, H5P_DEFAULT, H5P_DEFAULT);
    const char* cstr = s.c_str();
    H5Awrite(a, t, &cstr);
    H5Aclose(a);
    H5Sclose(sp);
    H5Tclose(t);
}

inline void WritePositionAttr(hid_t g, const std::vector<double>& pos,
                              int size_override) {
    int n = size_override > 0 ? size_override
                              : static_cast<int>(pos.size());
    hsize_t dims[1] = { static_cast<hsize_t>(n) };
    hid_t sp = H5Screate_simple(1, dims, nullptr);
    hid_t a = H5Acreate2(g, "position", H5T_NATIVE_DOUBLE, sp,
                         H5P_DEFAULT, H5P_DEFAULT);
    std::vector<double> buf(n, 0.0);
    for (int i = 0; i < n && i < static_cast<int>(pos.size()); ++i) {
        buf[i] = pos[i];
    }
    H5Awrite(a, H5T_NATIVE_DOUBLE, buf.data());
    H5Aclose(a);
    H5Sclose(sp);
}

inline void Write1D(hid_t g, const char* name,
                    const std::vector<float>& buf, int n_samples,
                    bool compress) {
    hsize_t dims[1] = { static_cast<hsize_t>(n_samples) };
    hid_t sp = H5Screate_simple(1, dims, nullptr);
    hid_t p = H5Pcreate(H5P_DATASET_CREATE);
    if (compress) {
        hsize_t chunks[1] = { static_cast<hsize_t>(
            n_samples < 1024 ? n_samples : 1024) };
        H5Pset_chunk(p, 1, chunks);
        H5Pset_deflate(p, 4);
    }
    hid_t ds = H5Dcreate2(g, name, H5T_NATIVE_FLOAT, sp,
                          H5P_DEFAULT, p, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Pclose(p);
    H5Sclose(sp);
}

inline void Write2D(hid_t g, const char* name,
                    const std::vector<float>& buf,
                    int dim0, int n_samples, bool compress) {
    hsize_t dims[2] = { static_cast<hsize_t>(dim0),
                        static_cast<hsize_t>(n_samples) };
    hid_t sp = H5Screate_simple(2, dims, nullptr);
    hid_t p = H5Pcreate(H5P_DATASET_CREATE);
    if (compress) {
        hsize_t chunks[2] = { static_cast<hsize_t>(dim0),
                              static_cast<hsize_t>(
                                  n_samples < 1024 ? n_samples : 1024) };
        H5Pset_chunk(p, 2, chunks);
        H5Pset_deflate(p, 4);
    }
    hid_t ds = H5Dcreate2(g, name, H5T_NATIVE_FLOAT, sp,
                          H5P_DEFAULT, p, H5P_DEFAULT);
    H5Dwrite(ds, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Pclose(p);
    H5Sclose(sp);
}

inline void WriteFixture(const std::string& path, const FixtureSpec& spec) {
    hid_t f = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    if (!spec.skip_format_version) {
        const std::string& fv = spec.override_format_version.empty()
            ? spec.format_version : spec.override_format_version;
        WriteStringAttr(f, "format_version", fv);
    }
    if (!spec.skip_dt) {
        WriteScalarAttr(f, "dt", H5T_NATIVE_DOUBLE, &spec.dt);
    }
    if (!spec.skip_t0) {
        WriteScalarAttr(f, "t0", H5T_NATIVE_DOUBLE, &spec.t0);
    }
    if (!spec.skip_n_samples) {
        int64_t ns = spec.n_samples;
        WriteScalarAttr(f, "n_samples", H5T_NATIVE_INT64, &ns);
    }
    if (!spec.skip_space_dim) {
        int sd = spec.space_dim;
        WriteScalarAttr(f, "space_dim", H5T_NATIVE_INT32, &sd);
    }

    // /shots/<NNNN>/receivers/...
    hid_t gshots = H5Gcreate2(f, "/shots", H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT);
    char shot_key[8];
    std::snprintf(shot_key, sizeof(shot_key), "%04d", spec.shot_id);
    hid_t gshot = H5Gcreate2(gshots, shot_key, H5P_DEFAULT, H5P_DEFAULT,
                             H5P_DEFAULT);
    {
        int sid = spec.shot_id;
        WriteScalarAttr(gshot, "shot_id", H5T_NATIVE_INT32, &sid);
    }
    hid_t grecv = H5Gcreate2(gshot, "receivers", H5P_DEFAULT, H5P_DEFAULT,
                             H5P_DEFAULT);
    for (const auto& r : spec.receivers) {
        hid_t rg = H5Gcreate2(grecv, r.name.c_str(), H5P_DEFAULT, H5P_DEFAULT,
                              H5P_DEFAULT);
        WritePositionAttr(rg, r.position, spec.override_position_size);

        auto write_vec = [&](const char* name,
                             const std::vector<float>& d,
                             const std::vector<float>& w) {
            if (d.empty()) return;
            const int d0 = spec.override_vector_dim0 > 0
                ? spec.override_vector_dim0 : spec.space_dim;
            Write2D(rg, name, d, d0, spec.n_samples, spec.compress);
            if (!w.empty()) {
                std::string wn = std::string("weight_") + name;
                Write2D(rg, wn.c_str(), w, d0, spec.n_samples, spec.compress);
            }
        };
        if (!r.pressure.empty()) {
            Write1D(rg, "PS", r.pressure, spec.n_samples, spec.compress);
            if (!r.pressure_weight.empty()) {
                int ns = spec.override_weight_nsamples > 0
                    ? spec.override_weight_nsamples : spec.n_samples;
                Write1D(rg, "weight_PS", r.pressure_weight, ns,
                        spec.compress);
            }
        }
        write_vec("VEL",  r.velocity,     r.velocity_weight);
        write_vec("DISP", r.displacement, r.displacement_weight);
        write_vec("ACC",  r.acceleration, r.acceleration_weight);
        H5Gclose(rg);
    }
    H5Gclose(grecv);
    H5Gclose(gshot);
    H5Gclose(gshots);
    H5Fclose(f);
}

inline FixtureSpec MakeBasic2DSpec(int n_samples = 8) {
    FixtureSpec s;
    s.space_dim = 2;
    s.n_samples = n_samples;
    s.dt = 1e-3;
    s.t0 = 0.0;
    auto linspace_vec = [&](float seed, int n) {
        std::vector<float> v(n);
        for (int i = 0; i < n; ++i) v[i] = seed + 0.125f * i;
        return v;
    };

    ReceiverSpec r1;
    r1.name = "R0001";
    r1.position = {10.0, 20.0};
    r1.pressure = linspace_vec(1.0f, n_samples);
    r1.velocity.resize(2 * n_samples);
    for (int c = 0; c < 2; ++c) {
        for (int i = 0; i < n_samples; ++i) {
            r1.velocity[c * n_samples + i] = 100.0f * (c + 1) + i;
        }
    }
    s.receivers.push_back(r1);

    ReceiverSpec r2;
    r2.name = "R0002";
    r2.position = {30.0, 40.0};
    r2.velocity.resize(2 * n_samples);
    for (int c = 0; c < 2; ++c) {
        for (int i = 0; i < n_samples; ++i) {
            r2.velocity[c * n_samples + i] = -5.0f * (c + 1) + i * 0.5f;
        }
    }
    s.receivers.push_back(r2);

    ReceiverSpec r3;
    r3.name = "R0003";
    r3.position = {50.0, 60.0};
    r3.pressure = linspace_vec(7.0f, n_samples);
    r3.pressure_weight.resize(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        r3.pressure_weight[i] = 0.25f + 0.1f * i;
    }
    s.receivers.push_back(r3);

    return s;
}

struct PathGuard {
    std::string path;
    ~PathGuard() { if (!path.empty()) std::remove(path.c_str()); }
};

}  // namespace sem_test

#endif  // SEM_TEST_HDF5_FIXTURE_WRITER_HPP
