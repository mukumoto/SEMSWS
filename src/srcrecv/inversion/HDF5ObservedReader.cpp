/**
 * @file HDF5ObservedReader.cpp
 * @brief HDF5 observed-data reader (catalog + per-rank hyperslab).
 *
 * Uses the HDF5 C API only (hdf5.h). The SEMSWS build environment has a
 * latent mismatch between the system-installed HDF5 C++ bindings (1.10 in
 * /usr/include/hdf5/serial) and the MFEM-linked C runtime (1.14 at
 * /home/kota/program_ubuntu/hdf5-1.14). The 1.14 install ships C headers
 * only, so sticking to the C API guarantees the headers we compile against
 * match the library we link/run against.
 */

#include "srcrecv/HDF5ObservedReader.hpp"
#include "srcrecv/HDF5IOSchema.hpp"
#include "srcrecv/ObservedTypes.hpp"

#include <hdf5.h>
#include <mfem.hpp>
#include <cstring>
#include <string>
#include <vector>

namespace SEM {

namespace {

using HDF5Schema::kFormatVersionMajor;

// Scoped HDF5 ID holder.
struct H5Id {
    hid_t id = -1;
    enum Kind { File, Group, Dataset, Attr, Space, Type, Plist, None } kind = None;
    H5Id() = default;
    H5Id(hid_t i, Kind k) : id(i), kind(k) {}
    H5Id(const H5Id&) = delete;
    H5Id& operator=(const H5Id&) = delete;
    H5Id(H5Id&& o) noexcept : id(o.id), kind(o.kind) { o.id = -1; o.kind = None; }
    H5Id& operator=(H5Id&& o) noexcept {
        reset();
        id = o.id; kind = o.kind; o.id = -1; o.kind = None;
        return *this;
    }
    ~H5Id() { reset(); }
    void reset() {
        if (id < 0) return;
        switch (kind) {
            case File:    H5Fclose(id); break;
            case Group:   H5Gclose(id); break;
            case Dataset: H5Dclose(id); break;
            case Attr:    H5Aclose(id); break;
            case Space:   H5Sclose(id); break;
            case Type:    H5Tclose(id); break;
            case Plist:   H5Pclose(id); break;
            case None: break;
        }
        id = -1; kind = None;
    }
    operator hid_t() const { return id; }
};

[[noreturn]] void Abort(const std::string& msg) {
    MFEM_ABORT(msg);
}

bool AttrExists(hid_t loc, const char* name) {
    return H5Aexists(loc, name) > 0;
}

std::string ReadStringAttr(hid_t loc, const char* name,
                           const std::string& ctx) {
    if (!AttrExists(loc, name)) {
        Abort("HDF5 observed: missing required attribute '" +
              std::string(name) + "' on " + ctx);
    }
    H5Id attr(H5Aopen(loc, name, H5P_DEFAULT), H5Id::Attr);
    if (attr < 0) Abort("HDF5 observed: cannot open attr '" + std::string(name)
                        + "' on " + ctx);
    H5Id t(H5Aget_type(attr), H5Id::Type);
    if (H5Tget_class(t) != H5T_STRING) {
        Abort("HDF5 observed: attr '" + std::string(name) + "' on " + ctx
              + " is not a string");
    }
    if (H5Tis_variable_str(t) > 0) {
        char* buf = nullptr;
        H5Id mtype(H5Tcopy(H5T_C_S1), H5Id::Type);
        H5Tset_size(mtype, H5T_VARIABLE);
        H5Tset_cset(mtype, H5T_CSET_UTF8);
        if (H5Aread(attr, mtype, &buf) < 0 || buf == nullptr) {
            Abort("HDF5 observed: failed reading string attr '"
                  + std::string(name) + "' on " + ctx);
        }
        std::string out(buf);
        H5Dvlen_reclaim(mtype, H5Aget_space(attr), H5P_DEFAULT, &buf);
        return out;
    } else {
        size_t sz = H5Tget_size(t);
        std::string out(sz, '\0');
        if (H5Aread(attr, t, out.data()) < 0) {
            Abort("HDF5 observed: failed reading string attr '"
                  + std::string(name) + "' on " + ctx);
        }
        // Strip trailing NUL.
        while (!out.empty() && out.back() == '\0') out.pop_back();
        return out;
    }
}

double ReadDoubleAttr(hid_t loc, const char* name, const std::string& ctx) {
    if (!AttrExists(loc, name)) {
        Abort("HDF5 observed: missing required attribute '" +
              std::string(name) + "' on " + ctx);
    }
    H5Id a(H5Aopen(loc, name, H5P_DEFAULT), H5Id::Attr);
    double v = 0.0;
    if (H5Aread(a, H5T_NATIVE_DOUBLE, &v) < 0) {
        Abort("HDF5 observed: failed reading double attr '" +
              std::string(name) + "' on " + ctx);
    }
    return v;
}

int64_t ReadInt64Attr(hid_t loc, const char* name, const std::string& ctx) {
    if (!AttrExists(loc, name)) {
        Abort("HDF5 observed: missing required attribute '" +
              std::string(name) + "' on " + ctx);
    }
    H5Id a(H5Aopen(loc, name, H5P_DEFAULT), H5Id::Attr);
    int64_t v = 0;
    if (H5Aread(a, H5T_NATIVE_INT64, &v) < 0) {
        Abort("HDF5 observed: failed reading int64 attr '" +
              std::string(name) + "' on " + ctx);
    }
    return v;
}

int ReadInt32Attr(hid_t loc, const char* name, const std::string& ctx) {
    if (!AttrExists(loc, name)) {
        Abort("HDF5 observed: missing required attribute '" +
              std::string(name) + "' on " + ctx);
    }
    H5Id a(H5Aopen(loc, name, H5P_DEFAULT), H5Id::Attr);
    int v = 0;
    if (H5Aread(a, H5T_NATIVE_INT32, &v) < 0) {
        Abort("HDF5 observed: failed reading int32 attr '" +
              std::string(name) + "' on " + ctx);
    }
    return v;
}

void ReadPositionAttr(hid_t g, int space_dim, Vector& out,
                      const std::string& ctx) {
    if (!AttrExists(g, "position")) {
        Abort("HDF5 observed: missing 'position' attr on " + ctx);
    }
    H5Id a(H5Aopen(g, "position", H5P_DEFAULT), H5Id::Attr);
    H5Id sp(H5Aget_space(a), H5Id::Space);
    int ndims = H5Sget_simple_extent_ndims(sp);
    hsize_t dims[2] = {0, 0};
    if (ndims >= 1) H5Sget_simple_extent_dims(sp, dims, nullptr);
    hsize_t n = (ndims >= 1) ? dims[0] : 1;
    if (static_cast<int>(n) != space_dim) {
        Abort("HDF5 observed: 'position' on " + ctx +
              " has size " + std::to_string(n) +
              ", expected " + std::to_string(space_dim));
    }
    std::vector<double> buf(n);
    if (H5Aread(a, H5T_NATIVE_DOUBLE, buf.data()) < 0) {
        Abort("HDF5 observed: failed reading 'position' on " + ctx);
    }
    out.SetSize(space_dim);
    for (int i = 0; i < space_dim; ++i) out(i) = static_cast<real_t>(buf[i]);
}

struct ChannelProbe {
    const char* name;
    ReceiverType type;
};

const ChannelProbe kProbes[] = {
    {HDF5Schema::kChannelVelocity,     ReceiverType::Velocity},
    {HDF5Schema::kChannelDisplacement, ReceiverType::Displacement},
    {HDF5Schema::kChannelAcceleration, ReceiverType::Acceleration},
    {HDF5Schema::kChannelPressure,     ReceiverType::Pressure},
};

bool LinkExists(hid_t loc, const char* name) {
    return H5Lexists(loc, name, H5P_DEFAULT) > 0;
}

void ValidateChannelShape(hid_t ds, ReceiverType type,
                          int space_dim, int n_samples,
                          const std::string& ctx) {
    H5Id sp(H5Dget_space(ds), H5Id::Space);
    int ndims = H5Sget_simple_extent_ndims(sp);
    hsize_t dims[3] = {0, 0, 0};
    if (ndims > 0) H5Sget_simple_extent_dims(sp, dims, nullptr);
    if (IsScalarObservedType(type)) {
        if (ndims != 1 || static_cast<int>(dims[0]) != n_samples) {
            Abort("HDF5 observed: " + ctx +
                  " scalar channel shape mismatch (expect [" +
                  std::to_string(n_samples) + "])");
        }
    } else {
        if (ndims != 2 ||
            static_cast<int>(dims[0]) != space_dim ||
            static_cast<int>(dims[1]) != n_samples) {
            Abort("HDF5 observed: " + ctx +
                  " vector channel shape mismatch (expect [" +
                  std::to_string(space_dim) + "," +
                  std::to_string(n_samples) + "])");
        }
    }
}

// Lightweight error-stack suppression: H5E_BEGIN/END_TRY isn't available as a
// function, so we push a no-op walker over the default error stack.
struct ErrStackGuard {
    H5E_auto2_t old_func = nullptr;
    void* old_data = nullptr;
    ErrStackGuard() {
        H5Eget_auto2(H5E_DEFAULT, &old_func, &old_data);
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    }
    ~ErrStackGuard() {
        H5Eset_auto2(H5E_DEFAULT, old_func, old_data);
    }
};

}  // namespace

// ---------------------------------------------------------------------------
// ReadCatalog
// ---------------------------------------------------------------------------

HDF5ObservedCatalog
HDF5ObservedReader::ReadCatalog(const std::string& path,
                                int expected_space_dim,
                                int shot_id) {
    HDF5ObservedCatalog cat;
    ErrStackGuard eg;

    H5Id file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Id::File);
    if (file < 0) {
        Abort("HDF5 observed: cannot open file " + path);
    }

    cat.format_version = ReadStringAttr(file, HDF5Schema::kAttrFormatVersion,
                                        path);
    std::string major = cat.format_version.substr(
        0, cat.format_version.find('.'));
    if (major != kFormatVersionMajor) {
        Abort("HDF5 observed: unsupported format_version '" +
              cat.format_version + "' in " + path +
              " (expected major " + std::string(kFormatVersionMajor) +
              "). Run driver/tools/migrate_observed_v1_to_v2.py to upgrade.");
    }
    cat.dt        = ReadDoubleAttr(file, HDF5Schema::kAttrDt, path);
    cat.t0        = ReadDoubleAttr(file, HDF5Schema::kAttrT0, path);
    cat.n_samples = static_cast<int>(
        ReadInt64Attr(file, HDF5Schema::kAttrNSamples, path));
    cat.space_dim = ReadInt32Attr(file, HDF5Schema::kAttrSpaceDim, path);

    // F14: units must be "SI" if present. Missing attribute is treated as
    // "SI" for backward-compat with v2.0 fixtures that pre-date the writer
    // emitting it. Any other value is rejected.
    if (AttrExists(file, HDF5Schema::kAttrUnits)) {
        const std::string u = ReadStringAttr(file, HDF5Schema::kAttrUnits, path);
        if (u != HDF5Schema::kDefaultUnits) {
            Abort("HDF5 observed: '" + std::string(HDF5Schema::kAttrUnits) +
                  "'='" + u + "' in " + path + " — only '" +
                  HDF5Schema::kDefaultUnits + "' is supported (F14)");
        }
    }

    if (cat.space_dim != expected_space_dim) {
        Abort("HDF5 observed: file " + path +
              " has space_dim=" + std::to_string(cat.space_dim) +
              ", simulation Dim=" + std::to_string(expected_space_dim));
    }
    if (cat.n_samples <= 0) {
        Abort("HDF5 observed: file " + path +
              " has n_samples=" + std::to_string(cat.n_samples));
    }

    // Navigate /shots/<NNNN>/receivers/...
    const std::string shot_path = HDF5Schema::ShotGroupPath(shot_id);
    if (!LinkExists(file, HDF5Schema::kGroupShots)) {
        Abort("HDF5 observed: file " + path + " has no /" +
              HDF5Schema::kGroupShots + " group (v2.0 schema required)");
    }
    if (!LinkExists(file, shot_path.c_str())) {
        Abort("HDF5 observed: file " + path + " has no /" + shot_path +
              " group (shot_id=" + std::to_string(shot_id) + ")");
    }
    H5Id gshot(H5Gopen2(file, shot_path.c_str(), H5P_DEFAULT), H5Id::Group);
    if (gshot < 0) {
        Abort("HDF5 observed: cannot open /" + shot_path + " in " + path);
    }
    if (!LinkExists(gshot, HDF5Schema::kGroupReceivers)) {
        Abort("HDF5 observed: " + path + ":/" + shot_path + " has no " +
              HDF5Schema::kGroupReceivers + " group");
    }
    H5Id grecv(H5Gopen2(gshot, HDF5Schema::kGroupReceivers, H5P_DEFAULT),
               H5Id::Group);
    if (grecv < 0) {
        Abort("HDF5 observed: cannot open /" + shot_path + "/" +
              HDF5Schema::kGroupReceivers + " in " + path);
    }

    H5G_info_t ginfo{};
    H5Gget_info(grecv, &ginfo);
    cat.receivers.reserve(static_cast<size_t>(ginfo.nlinks));

    for (hsize_t i = 0; i < ginfo.nlinks; ++i) {
        char namebuf[256];
        ssize_t len = H5Lget_name_by_idx(grecv, ".", H5_INDEX_NAME, H5_ITER_INC,
                                         i, namebuf, sizeof(namebuf),
                                         H5P_DEFAULT);
        if (len <= 0) continue;
        std::string rname(namebuf, static_cast<size_t>(len));
        std::string ctx = path + ":/receivers/" + rname;

        H5Id g(H5Gopen2(grecv, rname.c_str(), H5P_DEFAULT), H5Id::Group);
        if (g < 0) Abort("HDF5 observed: cannot open " + ctx);

        HDF5ReceiverEntry entry;
        entry.name = rname;
        ReadPositionAttr(g, cat.space_dim, entry.position, ctx);

        // F13: validate that every dataset under this receiver group is
        // a recognised channel or weight_<channel>. Reject unknown names
        // (e.g. legacy "gradient" / "GRAD") so writers can't smuggle in
        // out-of-spec data.
        H5G_info_t recv_info{};
        H5Gget_info(g, &recv_info);
        for (hsize_t k = 0; k < recv_info.nlinks; ++k) {
            char childbuf[256];
            ssize_t clen = H5Lget_name_by_idx(g, ".", H5_INDEX_NAME,
                                              H5_ITER_INC, k, childbuf,
                                              sizeof(childbuf), H5P_DEFAULT);
            if (clen <= 0) continue;
            const std::string cname(childbuf, static_cast<size_t>(clen));
            bool is_known = false;
            for (const auto& p : kProbes) {
                if (cname == p.name ||
                    cname == std::string("weight_") + p.name) {
                    is_known = true; break;
                }
            }
            if (!is_known) {
                Abort("HDF5 observed: " + ctx + " has unknown dataset '" +
                      cname + "' (F13: only " +
                      std::string(HDF5Schema::kChannelPressure) + ", " +
                      HDF5Schema::kChannelVelocity + ", " +
                      HDF5Schema::kChannelDisplacement + ", " +
                      HDF5Schema::kChannelAcceleration + " and weight_<*> "
                      "are allowed)");
            }
        }

        for (const auto& probe : kProbes) {
            if (!LinkExists(g, probe.name)) continue;
            H5Id ds(H5Dopen2(g, probe.name, H5P_DEFAULT), H5Id::Dataset);
            if (ds < 0) {
                Abort("HDF5 observed: cannot open dataset " + ctx
                      + "/" + probe.name);
            }
            ValidateChannelShape(ds, probe.type, cat.space_dim, cat.n_samples,
                                 ctx + "/" + probe.name);
            HDF5ChannelDescriptor ch;
            ch.type = probe.type;
            std::string wname = std::string("weight_") + probe.name;
            ch.has_weight = LinkExists(g, wname.c_str());
            if (ch.has_weight) {
                H5Id wds(H5Dopen2(g, wname.c_str(), H5P_DEFAULT), H5Id::Dataset);
                if (wds < 0) {
                    Abort("HDF5 observed: cannot open " + ctx + "/" + wname);
                }
                ValidateChannelShape(wds, probe.type, cat.space_dim,
                                     cat.n_samples, ctx + "/" + wname);
            }
            entry.channels.push_back(ch);
        }
        if (entry.channels.empty()) {
            Abort("HDF5 observed: receiver " + ctx
                  + " has no recognized channel dataset");
        }
        cat.receivers.push_back(std::move(entry));
    }

    if (cat.receivers.empty()) {
        Abort("HDF5 observed: file " + path + " contains no receivers");
    }
    return cat;
}

// ---------------------------------------------------------------------------
// ReadOwnedChannels
// ---------------------------------------------------------------------------

std::vector<HDF5ObservedReader::OwnedChannelResult>
HDF5ObservedReader::ReadOwnedChannels(
    const std::string& path,
    const std::vector<OwnedChannelRequest>& requests,
    int shot_id) {

    std::vector<OwnedChannelResult> out(requests.size());
    if (requests.empty()) return out;

    ErrStackGuard eg;
    H5Id file(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Id::File);
    if (file < 0) Abort("HDF5 observed: cannot open file " + path);

    int n_samples = static_cast<int>(
        ReadInt64Attr(file, HDF5Schema::kAttrNSamples, path));
    int space_dim = ReadInt32Attr(file, HDF5Schema::kAttrSpaceDim, path);

    const std::string shot_path = HDF5Schema::ShotGroupPath(shot_id);
    H5Id gshot(H5Gopen2(file, shot_path.c_str(), H5P_DEFAULT), H5Id::Group);
    if (gshot < 0) {
        Abort("HDF5 observed: cannot open /" + shot_path + " in " + path);
    }
    H5Id grecv(H5Gopen2(gshot, HDF5Schema::kGroupReceivers, H5P_DEFAULT),
               H5Id::Group);
    if (grecv < 0) {
        Abort("HDF5 observed: no /" + shot_path + "/" +
              HDF5Schema::kGroupReceivers + " in " + path);
    }

    for (size_t i = 0; i < requests.size(); ++i) {
        const auto& req = requests[i];
        auto& res = out[i];
        const char* cname = ObservedChannelName(req.type);
        std::string ctx = path + ":/receivers/" + req.receiver_name
                          + "/" + cname;

        H5Id g(H5Gopen2(grecv, req.receiver_name.c_str(), H5P_DEFAULT),
               H5Id::Group);
        if (g < 0) Abort("HDF5 observed: cannot open receiver group " + ctx);

        H5Id ds(H5Dopen2(g, cname, H5P_DEFAULT), H5Id::Dataset);
        if (ds < 0) Abort("HDF5 observed: cannot open " + ctx);
        H5Id fsp(H5Dget_space(ds), H5Id::Space);

        res.data.assign(n_samples, 0.0f);

        if (IsScalarObservedType(req.type)) {
            if (req.component != -1) {
                Abort("HDF5 observed: scalar channel " + ctx +
                      " requested with component=" + std::to_string(req.component));
            }
            hsize_t mdims[1] = { static_cast<hsize_t>(n_samples) };
            H5Id msp(H5Screate_simple(1, mdims, nullptr), H5Id::Space);
            if (H5Dread(ds, H5T_NATIVE_FLOAT, msp, fsp, H5P_DEFAULT,
                        res.data.data()) < 0) {
                Abort("HDF5 observed: read failure on " + ctx);
            }
        } else {
            if (req.component < 0 || req.component >= space_dim) {
                Abort("HDF5 observed: vector channel " + ctx +
                      " requested with out-of-range component=" +
                      std::to_string(req.component));
            }
            hsize_t offset[2] = { static_cast<hsize_t>(req.component), 0 };
            hsize_t count[2]  = { 1, static_cast<hsize_t>(n_samples) };
            if (H5Sselect_hyperslab(fsp, H5S_SELECT_SET, offset, nullptr,
                                    count, nullptr) < 0) {
                Abort("HDF5 observed: hyperslab select failed on " + ctx);
            }
            hsize_t mdims[1] = { static_cast<hsize_t>(n_samples) };
            H5Id msp(H5Screate_simple(1, mdims, nullptr), H5Id::Space);
            if (H5Dread(ds, H5T_NATIVE_FLOAT, msp, fsp, H5P_DEFAULT,
                        res.data.data()) < 0) {
                Abort("HDF5 observed: hyperslab read failure on " + ctx);
            }
        }

        if (req.want_weight) {
            std::string wname = std::string("weight_") + cname;
            if (H5Lexists(g, wname.c_str(), H5P_DEFAULT) > 0) {
                res.has_weight = true;
                res.weight.assign(n_samples, 0.0f);
                H5Id wds(H5Dopen2(g, wname.c_str(), H5P_DEFAULT),
                         H5Id::Dataset);
                if (wds < 0) {
                    Abort("HDF5 observed: cannot open " + ctx + " weight");
                }
                H5Id wfsp(H5Dget_space(wds), H5Id::Space);
                if (IsScalarObservedType(req.type)) {
                    hsize_t mdims[1] = { static_cast<hsize_t>(n_samples) };
                    H5Id msp(H5Screate_simple(1, mdims, nullptr), H5Id::Space);
                    if (H5Dread(wds, H5T_NATIVE_FLOAT, msp, wfsp, H5P_DEFAULT,
                                res.weight.data()) < 0) {
                        Abort("HDF5 observed: weight read failure on " + ctx);
                    }
                } else {
                    hsize_t offset[2] = {
                        static_cast<hsize_t>(req.component), 0 };
                    hsize_t count[2]  = { 1, static_cast<hsize_t>(n_samples) };
                    if (H5Sselect_hyperslab(wfsp, H5S_SELECT_SET, offset,
                                            nullptr, count, nullptr) < 0) {
                        Abort("HDF5 observed: weight hyperslab select failed on "
                              + ctx);
                    }
                    hsize_t mdims[1] = { static_cast<hsize_t>(n_samples) };
                    H5Id msp(H5Screate_simple(1, mdims, nullptr), H5Id::Space);
                    if (H5Dread(wds, H5T_NATIVE_FLOAT, msp, wfsp, H5P_DEFAULT,
                                res.weight.data()) < 0) {
                        Abort("HDF5 observed: weight hyperslab read failure on "
                              + ctx);
                    }
                }
            }
        }
    }
    return out;
}

}  // namespace SEM
