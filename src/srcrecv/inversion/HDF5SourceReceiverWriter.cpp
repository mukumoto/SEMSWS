// HDF5SourceReceiverWriter — source-side writer that emits
// /shots/<NNNN>/sources/ for a per-shot output HDF5 file.
// Plugs into ReceiverArray::SaveToHDF5.

#include "srcrecv/HDF5SourceReceiverWriter.hpp"
#include "srcrecv/HDF5IOSchema.hpp"
#include "srcrecv/Source.hpp"

#include <hdf5.h>
#include <mfem.hpp>
#include <cstring>
#include <string>

namespace SEM {

namespace {

void WriteScalarAttr(hid_t loc, const char* name, hid_t type,
                     const void* v) {
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
    hsize_t d[1] = { static_cast<hsize_t>(vs.size()) };
    hid_t sp = H5Screate_simple(1, d, nullptr);
    hid_t a = H5Acreate2(loc, name, t, sp, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<const char*> cstrs(vs.size());
    for (size_t i = 0; i < vs.size(); ++i) cstrs[i] = vs[i].c_str();
    H5Awrite(a, t, cstrs.data());
    H5Aclose(a); H5Sclose(sp); H5Tclose(t);
}

void WriteVecAttrF64(hid_t loc, const char* name,
                     const std::vector<real_t>& v) {
    hsize_t d[1] = { static_cast<hsize_t>(v.size()) };
    hid_t sp = H5Screate_simple(1, d, nullptr);
    hid_t a = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, sp,
                         H5P_DEFAULT, H5P_DEFAULT);
    std::vector<double> buf(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        buf[i] = static_cast<double>(v[i]);
    }
    H5Awrite(a, H5T_NATIVE_DOUBLE, buf.data());
    H5Aclose(a); H5Sclose(sp);
}

void WriteScalarStfDataset(hid_t loc, const std::vector<real_t>& stf) {
    hsize_t d[1] = { static_cast<hsize_t>(stf.size()) };
    hid_t sp = H5Screate_simple(1, d, nullptr);
    hid_t ds = H5Dcreate2(loc, HDF5Schema::kDatasetStf,
                          H5T_NATIVE_DOUBLE, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<double> buf(stf.size());
    for (size_t i = 0; i < stf.size(); ++i) {
        buf[i] = static_cast<double>(stf[i]);
    }
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Dclose(ds);
    H5Sclose(sp);
}

void WriteMomentTensorDataset(hid_t loc, int space_dim,
                              const std::vector<real_t>& mt) {
    const int expected_n = (space_dim == 3) ? 6 : 3;
    MFEM_VERIFY(static_cast<int>(mt.size()) == expected_n,
                "moment_tensor write: size mismatch (got " << mt.size()
                << ", expected " << expected_n << ")");
    hsize_t d[1] = { static_cast<hsize_t>(expected_n) };
    hid_t sp = H5Screate_simple(1, d, nullptr);
    hid_t ds = H5Dcreate2(loc, HDF5Schema::kDatasetMomentTensor,
                          H5T_NATIVE_DOUBLE, sp,
                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    std::vector<double> buf(expected_n);
    for (int i = 0; i < expected_n; ++i) buf[i] = static_cast<double>(mt[i]);
    H5Dwrite(ds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             buf.data());

    // Component order — canonical SEMSWS layout.
    static const std::vector<std::string> kCanon3D =
        {"Mxx", "Myy", "Mzz", "Mxy", "Mxz", "Myz"};
    static const std::vector<std::string> kCanon2D =
        {"Mxx", "Myy", "Mxy"};
    WriteStrListAttr(ds, HDF5Schema::kAttrComponentOrder,
                     space_dim == 3 ? kCanon3D : kCanon2D);
    WriteStrAttr(ds, HDF5Schema::kAttrCoordSystem, "xyz");

    H5Dclose(ds);
    H5Sclose(sp);
}

template <typename ConfigT>
std::vector<HDF5SourceWriteEntry>
BuildEntries(const ConfigT& cfg, int n_samples, real_t dt, int space_dim) {
    std::vector<HDF5SourceWriteEntry> out;
    out.reserve(cfg.forces.size() + cfg.pressures.size() +
                cfg.moment_tensors.size());

    auto stf_from_wavelet = [&](const SourceConfig::WaveletConfig& wv) {
        SourceTimeFunction tf =
            SourceTimeFunction::FromConfig(wv, n_samples, dt);
        std::vector<real_t> samples(static_cast<size_t>(n_samples));
        const mfem::DenseMatrix& m = tf.Data();
        // FromConfig always returns a 1-column matrix here (scalar layout).
        for (int i = 0; i < n_samples; ++i) {
            samples[i] = m(i, 0);
        }
        return samples;
    };

    auto vec_to_real = [](const std::vector<real_t>& v) { return v; };

    for (const auto& src : cfg.forces) {
        HDF5SourceWriteEntry e;
        e.id = src.id;
        e.type = HDF5Schema::kSourceTypeForce;
        e.position.assign(src.location.begin(), src.location.end());
        e.position.resize(space_dim, real_t{0});
        e.direction.assign(src.direction.begin(), src.direction.end());
        e.direction.resize(space_dim, real_t{0});
        e.stf = stf_from_wavelet(src.wavelet);
        out.push_back(std::move(e));
    }
    for (const auto& src : cfg.pressures) {
        HDF5SourceWriteEntry e;
        e.id = src.id;
        e.type = HDF5Schema::kSourceTypePressure;
        e.position.assign(src.location.begin(), src.location.end());
        e.position.resize(space_dim, real_t{0});
        e.stf = stf_from_wavelet(src.wavelet);
        out.push_back(std::move(e));
    }
    (void)vec_to_real;
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// 2D Build
// ---------------------------------------------------------------------------

std::vector<HDF5SourceWriteEntry>
BuildSourceWriteEntries(const SourceConfig::Config2D& cfg,
                        int n_samples, real_t dt) {
    auto out = BuildEntries(cfg, n_samples, dt, /*space_dim=*/2);
    // Append moment_tensor entries (template form had to skip them because
    // the field types differ between Config2D and Config3D).
    for (const auto& src : cfg.moment_tensors) {
        HDF5SourceWriteEntry e;
        e.id = src.id;
        e.type = HDF5Schema::kSourceTypeMomentTensor;
        e.position.assign(src.location.begin(), src.location.end());
        e.position.resize(2, real_t{0});
        // Canonical 2D order: {Mxx, Myy, Mxy}
        e.moment_tensor = {src.Mxx, src.Myy, src.Mxy};
        SourceTimeFunction tf =
            SourceTimeFunction::FromConfig(src.wavelet, n_samples, dt);
        e.stf.resize(static_cast<size_t>(n_samples));
        const mfem::DenseMatrix& m = tf.Data();
        for (int i = 0; i < n_samples; ++i) e.stf[i] = m(i, 0);
        out.push_back(std::move(e));
    }
    return out;
}

// ---------------------------------------------------------------------------
// 3D Build
// ---------------------------------------------------------------------------

std::vector<HDF5SourceWriteEntry>
BuildSourceWriteEntries(const SourceConfig::Config3D& cfg,
                        int n_samples, real_t dt) {
    auto out = BuildEntries(cfg, n_samples, dt, /*space_dim=*/3);
    for (const auto& src : cfg.moment_tensors) {
        HDF5SourceWriteEntry e;
        e.id = src.id;
        e.type = HDF5Schema::kSourceTypeMomentTensor;
        e.position.assign(src.location.begin(), src.location.end());
        e.position.resize(3, real_t{0});
        // Canonical 3D order: {Mxx, Myy, Mzz, Mxy, Mxz, Myz}
        e.moment_tensor = {src.Mxx, src.Myy, src.Mzz,
                           src.Mxy, src.Mxz, src.Myz};
        SourceTimeFunction tf =
            SourceTimeFunction::FromConfig(src.wavelet, n_samples, dt);
        e.stf.resize(static_cast<size_t>(n_samples));
        const mfem::DenseMatrix& m = tf.Data();
        for (int i = 0; i < n_samples; ++i) e.stf[i] = m(i, 0);
        out.push_back(std::move(e));
    }
    return out;
}

// ---------------------------------------------------------------------------
// WriteSourcesIntoShotGroup — rank-0 caller emits /sources/S<id>/...
// ---------------------------------------------------------------------------

void WriteSourcesIntoShotGroup(
    hid_t shot_group,
    int space_dim,
    const std::vector<HDF5SourceWriteEntry>& entries) {

    if (entries.empty()) return;

    hid_t g_src = H5Gcreate2(shot_group, HDF5Schema::kGroupSources,
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    MFEM_VERIFY(g_src >= 0, "WriteSourcesIntoShotGroup: cannot create "
                "/sources subgroup");

    for (const auto& e : entries) {
        const std::string key = HDF5Schema::SourceKey(e.id);
        hid_t g = H5Gcreate2(g_src, key.c_str(),
                             H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        MFEM_VERIFY(g >= 0, "cannot create source group " << key);

        int id_v = e.id;
        WriteScalarAttr(g, HDF5Schema::kAttrId, H5T_NATIVE_INT32, &id_v);
        if (!e.label.empty()) {
            WriteStrAttr(g, HDF5Schema::kAttrLabel, e.label);
        }
        WriteStrAttr(g, HDF5Schema::kAttrType, e.type);
        WriteVecAttrF64(g, HDF5Schema::kAttrPosition, e.position);
        if (e.type == HDF5Schema::kSourceTypeForce) {
            WriteVecAttrF64(g, HDF5Schema::kAttrDirection, e.direction);
        }
        if (e.type == HDF5Schema::kSourceTypeMomentTensor) {
            WriteMomentTensorDataset(g, space_dim, e.moment_tensor);
        }
        WriteScalarStfDataset(g, e.stf);

        H5Gclose(g);
    }
    H5Gclose(g_src);
}

}  // namespace SEM
