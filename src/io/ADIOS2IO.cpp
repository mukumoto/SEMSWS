/**
 * @file ADIOS2IO.cpp
 * @brief ADIOS2-based parallel I/O for MaterialField and L-BFGS history
 */

#include "io/ADIOS2IO.hpp"
#include <adios2.h>
#include <stdexcept>
#include <sstream>

namespace SEM {

// =============================================================================
// Helper: ADIOS2 type for real_t
// =============================================================================

// MFEM's real_t is either float or double depending on build config.
// ADIOS2 needs the concrete type for DefineVariable.

// =============================================================================
// SaveFieldBP
// =============================================================================

void SaveFieldBP(const std::string& filename,
                 const std::string& var_name,
                 const MaterialField& field,
                 const std::string& mesh_file,
                 MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    const int ne_local = field.NumElements();
    const int ngllx = field.NumGLLx();
    const int nglly = field.NumGLLy();
    const int local_size = ne_local * nglly * ngllx;

    // Gather ne_global via MPI_Allreduce
    int ne_global = 0;
    MPI_Allreduce(&ne_local, &ne_global, 1, MPI_INT, MPI_SUM, comm);

    // Compute this rank's offset in global element ordering
    int ne_offset = 0;
    MPI_Exscan(&ne_local, &ne_offset, 1, MPI_INT, MPI_SUM, comm);
    // MPI_Exscan leaves rank 0 undefined
    if (rank == 0) ne_offset = 0;

    const size_t global_size = static_cast<size_t>(ne_global) * nglly * ngllx;
    const size_t offset = static_cast<size_t>(ne_offset) * nglly * ngllx;

    // Create ADIOS2 instance
    adios2::ADIOS adios(comm);
    adios2::IO io = adios.DeclareIO("FieldIO");
    io.SetEngine("BP5");

    // Define variable with global shape
    adios2::Variable<real_t> var = io.DefineVariable<real_t>(
        var_name,
        {global_size},           // global shape
        {offset},                // local offset
        {static_cast<size_t>(local_size)}  // local count
    );

    // Store metadata as attributes
    io.DefineAttribute<int>("ne_global", ne_global);
    io.DefineAttribute<int>("ngllx", ngllx);
    io.DefineAttribute<int>("nglly", nglly);
    io.DefineAttribute<int>("nprocs", nprocs);
    io.DefineAttribute<std::string>("mesh_file", mesh_file);

    // Store per-rank ne_local so LoadFieldBP can reconstruct the partition
    adios2::Variable<int> var_ne_local = io.DefineVariable<int>(
        "ne_local",
        {static_cast<size_t>(nprocs)},
        {static_cast<size_t>(rank)},
        {1}
    );

    // Open, write, close
    adios2::Engine engine = io.Open(filename, adios2::Mode::Write);
    engine.BeginStep();

    engine.Put(var_ne_local, &ne_local);

    // Get host-readable pointer to field data
    const real_t* data_ptr = field.Data().HostRead();
    engine.Put(var, data_ptr);

    engine.EndStep();
    engine.Close();
}

// =============================================================================
// SaveFieldBP — 3D overload (extra ngllz attribute, ngllz*nglly*ngllx per elem)
// =============================================================================

void SaveFieldBP(const std::string& filename,
                 const std::string& var_name,
                 const MaterialField3D& field,
                 const std::string& mesh_file,
                 MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    const int ne_local = field.NumElements();
    const int ngllx = field.NumGLLx();
    const int nglly = field.NumGLLy();
    const int ngllz = field.NumGLLz();
    const int per_elem = ngllz * nglly * ngllx;
    const int local_size = ne_local * per_elem;

    int ne_global = 0;
    MPI_Allreduce(&ne_local, &ne_global, 1, MPI_INT, MPI_SUM, comm);

    int ne_offset = 0;
    MPI_Exscan(&ne_local, &ne_offset, 1, MPI_INT, MPI_SUM, comm);
    if (rank == 0) ne_offset = 0;

    const size_t global_size = static_cast<size_t>(ne_global) * per_elem;
    const size_t offset = static_cast<size_t>(ne_offset) * per_elem;

    adios2::ADIOS adios(comm);
    adios2::IO io = adios.DeclareIO("FieldIO");
    io.SetEngine("BP5");

    adios2::Variable<real_t> var = io.DefineVariable<real_t>(
        var_name,
        {global_size},
        {offset},
        {static_cast<size_t>(local_size)}
    );

    io.DefineAttribute<int>("ne_global", ne_global);
    io.DefineAttribute<int>("ngllx", ngllx);
    io.DefineAttribute<int>("nglly", nglly);
    io.DefineAttribute<int>("ngllz", ngllz);
    io.DefineAttribute<int>("nprocs", nprocs);
    io.DefineAttribute<std::string>("mesh_file", mesh_file);

    adios2::Variable<int> var_ne_local = io.DefineVariable<int>(
        "ne_local",
        {static_cast<size_t>(nprocs)},
        {static_cast<size_t>(rank)},
        {1}
    );

    adios2::Engine engine = io.Open(filename, adios2::Mode::Write);
    engine.BeginStep();

    engine.Put(var_ne_local, &ne_local);
    const real_t* data_ptr = field.Data().HostRead();
    engine.Put(var, data_ptr);

    engine.EndStep();
    engine.Close();
}

// =============================================================================
// LoadFieldBP
// =============================================================================

MaterialField LoadFieldBP(const std::string& filename,
                          const std::string& var_name,
                          MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    adios2::ADIOS adios(comm);
    adios2::IO io = adios.DeclareIO("FieldIO");
    io.SetEngine("BP5");

    adios2::Engine engine = io.Open(filename, adios2::Mode::Read);
    engine.BeginStep();

    // Read metadata attributes
    auto attr_ngllx = io.InquireAttribute<int>("ngllx");
    auto attr_nglly = io.InquireAttribute<int>("nglly");
    auto attr_nprocs = io.InquireAttribute<int>("nprocs");

    if (!attr_ngllx || !attr_nglly || !attr_nprocs) {
        throw std::runtime_error("ADIOS2IO: Missing required attributes in " + filename);
    }

    const int ngllx = attr_ngllx.Data()[0];
    const int nglly = attr_nglly.Data()[0];
    const int file_nprocs = attr_nprocs.Data()[0];

    MFEM_VERIFY(nprocs == file_nprocs,
        "ADIOS2IO: File " << filename << " was written with " << file_nprocs
        << " ranks but loading with " << nprocs);

    // Read this rank's ne_local from the stored partition info
    auto var_ne_local = io.InquireVariable<int>("ne_local");
    if (!var_ne_local) {
        throw std::runtime_error("ADIOS2IO: Variable 'ne_local' not found in " + filename);
    }
    var_ne_local.SetSelection({{static_cast<size_t>(rank)}, {1}});
    int ne_local = 0;
    engine.Get(var_ne_local, &ne_local, adios2::Mode::Sync);

    // Compute offset by reading all ne_local values up to this rank
    // We read all ranks' ne_local and sum up to rank-1
    std::vector<int> all_ne_local(nprocs);
    var_ne_local.SetSelection({{0}, {static_cast<size_t>(nprocs)}});
    engine.Get(var_ne_local, all_ne_local.data(), adios2::Mode::Sync);

    int ne_offset = 0;
    for (int r = 0; r < rank; r++) {
        ne_offset += all_ne_local[r];
    }

    const size_t local_size = static_cast<size_t>(ne_local) * nglly * ngllx;
    const size_t offset = static_cast<size_t>(ne_offset) * nglly * ngllx;

    // Inquire data variable
    auto var = io.InquireVariable<real_t>(var_name);
    if (!var) {
        throw std::runtime_error("ADIOS2IO: Variable '" + var_name + "' not found in " + filename);
    }

    var.SetSelection({{offset}, {local_size}});

    // Create MaterialField and read into it
    MaterialField field(ne_local, ngllx, nglly);
    real_t* data_ptr = field.Data().HostWrite();
    engine.Get(var, data_ptr);

    engine.EndStep();
    engine.Close();

    return field;
}

// =============================================================================
// LoadFieldBP3D
// =============================================================================

MaterialField3D LoadFieldBP3D(const std::string& filename,
                              const std::string& var_name,
                              MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    adios2::ADIOS adios(comm);
    adios2::IO io = adios.DeclareIO("FieldIO");
    io.SetEngine("BP5");

    adios2::Engine engine = io.Open(filename, adios2::Mode::Read);
    engine.BeginStep();

    auto attr_ngllx = io.InquireAttribute<int>("ngllx");
    auto attr_nglly = io.InquireAttribute<int>("nglly");
    auto attr_ngllz = io.InquireAttribute<int>("ngllz");
    auto attr_nprocs = io.InquireAttribute<int>("nprocs");

    if (!attr_ngllx || !attr_nglly || !attr_ngllz || !attr_nprocs) {
        throw std::runtime_error(
            "ADIOS2IO: Missing required 3D attributes (ngllx/ngllx/ngllz/nprocs) in "
            + filename);
    }

    const int ngllx = attr_ngllx.Data()[0];
    const int nglly = attr_nglly.Data()[0];
    const int ngllz = attr_ngllz.Data()[0];
    const int file_nprocs = attr_nprocs.Data()[0];

    MFEM_VERIFY(nprocs == file_nprocs,
        "ADIOS2IO: File " << filename << " was written with " << file_nprocs
        << " ranks but loading with " << nprocs);

    auto var_ne_local = io.InquireVariable<int>("ne_local");
    if (!var_ne_local) {
        throw std::runtime_error("ADIOS2IO: Variable 'ne_local' not found in " + filename);
    }
    var_ne_local.SetSelection({{static_cast<size_t>(rank)}, {1}});
    int ne_local = 0;
    engine.Get(var_ne_local, &ne_local, adios2::Mode::Sync);

    std::vector<int> all_ne_local(nprocs);
    var_ne_local.SetSelection({{0}, {static_cast<size_t>(nprocs)}});
    engine.Get(var_ne_local, all_ne_local.data(), adios2::Mode::Sync);
    int ne_offset = 0;
    for (int r = 0; r < rank; r++) ne_offset += all_ne_local[r];

    const int per_elem = ngllz * nglly * ngllx;
    const size_t local_size = static_cast<size_t>(ne_local) * per_elem;
    const size_t offset = static_cast<size_t>(ne_offset) * per_elem;

    auto var = io.InquireVariable<real_t>(var_name);
    if (!var) {
        throw std::runtime_error("ADIOS2IO: Variable '" + var_name + "' not found in " + filename);
    }
    var.SetSelection({{offset}, {local_size}});

    MaterialField3D field(ne_local, ngllx, nglly, ngllz);
    real_t* data_ptr = field.Data().HostWrite();
    engine.Get(var, data_ptr);

    engine.EndStep();
    engine.Close();

    return field;
}

// =============================================================================
// SaveLBFGSHistoryBP
// =============================================================================

void SaveLBFGSHistoryBP(const std::string& filename,
                        const std::vector<Vector>& s,
                        const std::vector<Vector>& y,
                        int iteration, MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if (s.size() != y.size()) {
        throw std::runtime_error("ADIOS2IO: s and y vectors must have same size");
    }

    const int num_pairs = static_cast<int>(s.size());

    adios2::ADIOS adios(comm);
    adios2::IO io = adios.DeclareIO("LBFGSHistoryIO");
    io.SetEngine("BP5");

    // Store metadata
    io.DefineAttribute<int>("iteration", iteration);
    io.DefineAttribute<int>("num_pairs", num_pairs);

    adios2::Engine engine = io.Open(filename, adios2::Mode::Write);
    engine.BeginStep();

    if (num_pairs > 0) {
        // All vectors should have the same local size
        const int local_size = s[0].Size();

        // Compute global size and offset
        int global_size = 0;
        MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);

        int local_offset = 0;
        MPI_Exscan(&local_size, &local_offset, 1, MPI_INT, MPI_SUM, comm);
        if (rank == 0) local_offset = 0;

        io.DefineAttribute<int>("vector_size_global", global_size);
        io.DefineAttribute<int>("nprocs", nprocs);

        // Store per-rank local sizes so LoadLBFGSHistoryBP can reconstruct partition
        adios2::Variable<int> var_local_size = io.DefineVariable<int>(
            "local_size",
            {static_cast<size_t>(nprocs)},
            {static_cast<size_t>(rank)},
            {1}
        );
        engine.Put(var_local_size, &local_size);

        for (int i = 0; i < num_pairs; i++) {
            std::string s_name = "s_" + std::to_string(i);
            std::string y_name = "y_" + std::to_string(i);

            auto var_s = io.DefineVariable<real_t>(
                s_name,
                {static_cast<size_t>(global_size)},
                {static_cast<size_t>(local_offset)},
                {static_cast<size_t>(local_size)}
            );
            auto var_y = io.DefineVariable<real_t>(
                y_name,
                {static_cast<size_t>(global_size)},
                {static_cast<size_t>(local_offset)},
                {static_cast<size_t>(local_size)}
            );

            engine.Put(var_s, s[i].HostRead());
            engine.Put(var_y, y[i].HostRead());
        }
    }

    engine.EndStep();
    engine.Close();
}

// =============================================================================
// LoadLBFGSHistoryBP
// =============================================================================

LBFGSHistory LoadLBFGSHistoryBP(const std::string& filename, MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    adios2::ADIOS adios(comm);
    adios2::IO io = adios.DeclareIO("LBFGSHistoryIO");
    io.SetEngine("BP5");

    adios2::Engine engine = io.Open(filename, adios2::Mode::Read);
    engine.BeginStep();

    auto attr_iter = io.InquireAttribute<int>("iteration");
    auto attr_pairs = io.InquireAttribute<int>("num_pairs");

    if (!attr_iter || !attr_pairs) {
        throw std::runtime_error("ADIOS2IO: Missing attributes in " + filename);
    }

    LBFGSHistory history;
    history.iteration = attr_iter.Data()[0];
    const int num_pairs = attr_pairs.Data()[0];

    if (num_pairs > 0) {
        auto attr_size = io.InquireAttribute<int>("vector_size_global");
        if (!attr_size) {
            throw std::runtime_error("ADIOS2IO: Missing vector_size_global in " + filename);
        }

        // Read per-rank local sizes stored by SaveLBFGSHistoryBP
        auto var_local_size = io.InquireVariable<int>("local_size");
        int local_size = 0;
        int local_offset = 0;

        if (var_local_size) {
            // New format: read stored partition
            auto attr_nprocs = io.InquireAttribute<int>("nprocs");
            MFEM_VERIFY(attr_nprocs,
                "ADIOS2IO: Missing nprocs attribute in " + filename);
            const int file_nprocs = attr_nprocs.Data()[0];
            MFEM_VERIFY(nprocs == file_nprocs,
                "ADIOS2IO: History file " << filename << " was written with "
                << file_nprocs << " ranks but loading with " << nprocs);

            std::vector<int> all_local_sizes(nprocs);
            var_local_size.SetSelection({{0}, {static_cast<size_t>(nprocs)}});
            engine.Get(var_local_size, all_local_sizes.data(), adios2::Mode::Sync);

            local_size = all_local_sizes[rank];
            for (int r = 0; r < rank; r++) {
                local_offset += all_local_sizes[r];
            }
        } else {
            // Legacy format: distribute evenly
            const int global_size = attr_size.Data()[0];
            const int per_rank = global_size / nprocs;
            const int remainder = global_size % nprocs;
            local_size = per_rank + (rank < remainder ? 1 : 0);
            if (rank < remainder) {
                local_offset = rank * (per_rank + 1);
            } else {
                local_offset = remainder * (per_rank + 1) + (rank - remainder) * per_rank;
            }
        }

        history.s.resize(num_pairs);
        history.y.resize(num_pairs);

        for (int i = 0; i < num_pairs; i++) {
            std::string s_name = "s_" + std::to_string(i);
            std::string y_name = "y_" + std::to_string(i);

            auto var_s = io.InquireVariable<real_t>(s_name);
            auto var_y = io.InquireVariable<real_t>(y_name);

            if (!var_s || !var_y) {
                throw std::runtime_error("ADIOS2IO: Missing variable " + s_name + " or " + y_name);
            }

            var_s.SetSelection({{static_cast<size_t>(local_offset)},
                                {static_cast<size_t>(local_size)}});
            var_y.SetSelection({{static_cast<size_t>(local_offset)},
                                {static_cast<size_t>(local_size)}});

            history.s[i].SetSize(local_size);
            history.y[i].SetSize(local_size);

            engine.Get(var_s, history.s[i].HostWrite());
            engine.Get(var_y, history.y[i].HostWrite());
        }
    }

    engine.EndStep();
    engine.Close();

    return history;
}

// =============================================================================
// SaveAdamStateBP
// =============================================================================

void SaveAdamStateBP(const std::string& filename,
                     const Vector& m, const Vector& v,
                     int iteration, MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    MFEM_VERIFY(m.Size() == v.Size(),
        "ADIOS2IO: Adam m and v must have same size");

    const int local_size = m.Size();

    adios2::ADIOS adios(comm);
    adios2::IO io = adios.DeclareIO("AdamStateIO");
    io.SetEngine("BP5");

    io.DefineAttribute<int>("iteration", iteration);
    io.DefineAttribute<int>("nprocs", nprocs);

    adios2::Engine engine = io.Open(filename, adios2::Mode::Write);
    engine.BeginStep();

    // Global size and offset
    int global_size = 0;
    MPI_Allreduce(&local_size, &global_size, 1, MPI_INT, MPI_SUM, comm);

    int local_offset = 0;
    MPI_Exscan(&local_size, &local_offset, 1, MPI_INT, MPI_SUM, comm);
    if (rank == 0) local_offset = 0;

    io.DefineAttribute<int>("vector_size_global", global_size);

    // Store per-rank local sizes
    auto var_local_size = io.DefineVariable<int>(
        "local_size",
        {static_cast<size_t>(nprocs)},
        {static_cast<size_t>(rank)},
        {1}
    );
    engine.Put(var_local_size, &local_size);

    // First moment
    auto var_m = io.DefineVariable<real_t>(
        "adam_m",
        {static_cast<size_t>(global_size)},
        {static_cast<size_t>(local_offset)},
        {static_cast<size_t>(local_size)}
    );
    engine.Put(var_m, m.HostRead());

    // Second moment
    auto var_v = io.DefineVariable<real_t>(
        "adam_v",
        {static_cast<size_t>(global_size)},
        {static_cast<size_t>(local_offset)},
        {static_cast<size_t>(local_size)}
    );
    engine.Put(var_v, v.HostRead());

    engine.EndStep();
    engine.Close();
}

// =============================================================================
// LoadAdamStateBP
// =============================================================================

AdamState LoadAdamStateBP(const std::string& filename, MPI_Comm comm) {
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    adios2::ADIOS adios(comm);
    adios2::IO io = adios.DeclareIO("AdamStateIO");
    io.SetEngine("BP5");

    adios2::Engine engine = io.Open(filename, adios2::Mode::Read);
    engine.BeginStep();

    auto attr_iter = io.InquireAttribute<int>("iteration");
    if (!attr_iter) {
        throw std::runtime_error("ADIOS2IO: Missing iteration in " + filename);
    }

    AdamState state;
    state.iteration = attr_iter.Data()[0];

    auto attr_nprocs = io.InquireAttribute<int>("nprocs");
    MFEM_VERIFY(attr_nprocs,
        "ADIOS2IO: Missing nprocs attribute in " + filename);
    const int file_nprocs = attr_nprocs.Data()[0];
    MFEM_VERIFY(nprocs == file_nprocs,
        "ADIOS2IO: Adam state " << filename << " was written with "
        << file_nprocs << " ranks but loading with " << nprocs);

    // Read per-rank local sizes
    auto var_local_size = io.InquireVariable<int>("local_size");
    MFEM_VERIFY(var_local_size,
        "ADIOS2IO: Missing local_size in " + filename);

    std::vector<int> all_local_sizes(nprocs);
    var_local_size.SetSelection({{0}, {static_cast<size_t>(nprocs)}});
    engine.Get(var_local_size, all_local_sizes.data(), adios2::Mode::Sync);

    int local_size = all_local_sizes[rank];
    int local_offset = 0;
    for (int r = 0; r < rank; r++) {
        local_offset += all_local_sizes[r];
    }

    // Read first moment
    auto var_m = io.InquireVariable<real_t>("adam_m");
    MFEM_VERIFY(var_m, "ADIOS2IO: Missing adam_m in " + filename);
    var_m.SetSelection({{static_cast<size_t>(local_offset)},
                        {static_cast<size_t>(local_size)}});
    state.m.SetSize(local_size);
    engine.Get(var_m, state.m.HostWrite());

    // Read second moment
    auto var_v = io.InquireVariable<real_t>("adam_v");
    MFEM_VERIFY(var_v, "ADIOS2IO: Missing adam_v in " + filename);
    var_v.SetSelection({{static_cast<size_t>(local_offset)},
                        {static_cast<size_t>(local_size)}});
    state.v.SetSize(local_size);
    engine.Get(var_v, state.v.HostWrite());

    engine.EndStep();
    engine.Close();

    return state;
}

}  // namespace SEM
