/**
 * @file coupling_probe_interface.cpp
 * @brief Phase 2A probe — build FluidSolidInterface and dump its caches.
 *
 * Loads a parent mesh, creates the fluid/solid ParSubMeshes with matching
 * scalar/vector FE spaces, calls FluidSolidInterface::Setup, and emits
 * KEY=VALUE diagnostic lines the pytest driver asserts on. Also exercises
 * the Extract* gather paths with a known analytical field to verify the
 * fluid/solid DOF lists line up with the cached normals and positions.
 */

#include <mfem.hpp>

#include "coupling/FluidSolidInterface.hpp"

#include <iomanip>
#include <iostream>
#include <string>

using namespace mfem;
using namespace SEM;

template<int Dim>
static int RunProbe(const std::string& mesh_file,
                    int order, int fluid_attr, int solid_attr)
{
    Mesh smesh(mesh_file.c_str(), 1, 1);
    ParMesh parent(MPI_COMM_WORLD, smesh);

    Array<int> fa(1); fa[0] = fluid_attr;
    Array<int> sa(1); sa[0] = solid_attr;
    ParSubMesh fluid_sub = ParSubMesh::CreateFromDomain(parent, fa);
    ParSubMesh solid_sub = ParSubMesh::CreateFromDomain(parent, sa);

    H1_FECollection fec(order, Dim);
    ParFiniteElementSpace fluid_fes(&fluid_sub, &fec, 1,
                                    mfem::Ordering::byNODES);
    ParFiniteElementSpace solid_fes(&solid_sub, &fec, Dim,
                                    mfem::Ordering::byNODES);

    FluidSolidInterface<Dim> iface;
    iface.Setup(parent, fluid_sub, solid_sub, fluid_attr, solid_attr,
                fluid_fes, solid_fes);

    const int nq_local = iface.NumLocalQuadPoints();
    const int nq_remote_fluid_owned = iface.NumRemoteFluidOwnedQuadPoints();
    const int nq_remote_solid_owned = iface.NumRemoteSolidOwnedQuadPoints();

    long long nq_total = 0;
    long long nq_local_ll = nq_local;
    MPI_Reduce(&nq_local_ll, &nq_total, 1, MPI_LONG_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);
    long long nq_remote_fl = nq_remote_fluid_owned;
    long long nq_remote_sl = nq_remote_solid_owned;
    long long nq_remote_fl_tot = 0, nq_remote_sl_tot = 0;
    MPI_Reduce(&nq_remote_fl, &nq_remote_fl_tot, 1, MPI_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&nq_remote_sl, &nq_remote_sl_tot, 1, MPI_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);

    // Normal direction check: average normal per rank, reduce sum, then
    // compare to ẑ (for the probe mesh the interface is at z = Lz/2, so the
    // fluid→solid normal should be +ẑ everywhere, average ~= (0,0,1)).
    Vector nsum(Dim);
    nsum = 0.0;
    const Vector& N = iface.Normals();
    for (int i = 0; i < nq_local; ++i) {
        for (int d = 0; d < Dim; ++d) nsum[d] += N[i * Dim + d];
    }
    std::vector<real_t> nsum_global(Dim, 0.0);
    MPI_Reduce(nsum.GetData(), nsum_global.data(), Dim,
               sizeof(real_t) == 4 ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    // Unit-length check local to this rank: max |1 - |n||.
    real_t max_dev = 0.0;
    for (int i = 0; i < nq_local; ++i) {
        real_t sq = 0;
        for (int d = 0; d < Dim; ++d) sq += N[i * Dim + d] * N[i * Dim + d];
        const real_t dev = std::fabs(1.0 - std::sqrt(sq));
        if (dev > max_dev) max_dev = dev;
    }
    real_t max_dev_global = 0.0;
    MPI_Reduce(&max_dev, &max_dev_global, 1,
               sizeof(real_t) == 4 ? MPI_FLOAT : MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // Test 2.2: analytical u_s = (0, 0, z). Fill solid displacement, call
    // ExtractSolidDisplacementNormal, and compare against expected z * n_z.
    ParGridFunction u_s(&solid_fes);
    u_s = 0.0;
    {
        // Last component (vdim=Dim-1) -> set equal to z at each scalar DOF.
        const int nscalar = solid_fes.GetNDofs();
        real_t* data = u_s.HostReadWrite();
        const ParMesh* smesh_ptr = solid_fes.GetParMesh();
        const int zc = Dim - 1;  // index of z component in 3D, y in 2D
        for (int i = 0; i < nscalar; ++i) {
            // DOF index for component zc when ordering==byNODES:
            //   vdof = i + zc * nscalar
            const int vdof = i + zc * nscalar;
            // For H1 Lagrange on GLL, scalar DOF i coincides with a mesh
            // "node" — for a simple H1 H-collection on a hex mesh that
            // lines up with a particular (element, local-node) pair. The
            // exact position is irrelevant for this test as long as the
            // value set at the fluid-DOF position is sampled back; we use
            // the submesh's GridFunction node coordinates via a temporary.
            data[vdof] = 0.0;  // default; overwritten below
        }
        // Set u_s to a linear field through the node-position grid function.
        ParGridFunction pos_gf(const_cast<ParFiniteElementSpace*>(&solid_fes));
        VectorFunctionCoefficient coef(Dim, [](const Vector& x, Vector& v) {
            v = 0.0;
            v[v.Size() - 1] = x[x.Size() - 1];  // u_z = z
        });
        u_s.ProjectCoefficient(coef);
    }
    Vector un(nq_local);
    iface.ExtractSolidDisplacementNormal(u_s, un);

    // Predicted un[i] = z_quad[i] * n_z[i]
    const Vector& pos = iface.QuadPositions();
    real_t max_err = 0.0;
    real_t max_abs_un = 0.0;
    for (int i = 0; i < nq_local; ++i) {
        const real_t z_i = pos[i * Dim + (Dim - 1)];
        const real_t nz  = N  [i * Dim + (Dim - 1)];
        const real_t pred = z_i * nz;
        const real_t err  = std::fabs(un[i] - pred);
        if (err > max_err)   max_err = err;
        if (std::fabs(un[i]) > max_abs_un) max_abs_un = std::fabs(un[i]);
    }
    real_t max_err_global = 0.0, max_abs_un_global = 0.0;
    MPI_Reduce(&max_err,    &max_err_global,    1,
               sizeof(real_t) == 4 ? MPI_FLOAT : MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&max_abs_un, &max_abs_un_global, 1,
               sizeof(real_t) == 4 ? MPI_FLOAT : MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    // Face-area consistency check: Σ jacobian_w_[i] across all quad points
    // should equal the physical interface area. For the probe mesh that's
    // Lx × Ly (planar interface at z = Lz/2).
    const Vector& jw = iface.JacobianTimesWeight();
    real_t jw_sum_local = 0.0;
    for (int i = 0; i < nq_local; ++i) jw_sum_local += jw[i];
    real_t jw_sum_global = 0.0;
    MPI_Reduce(&jw_sum_local, &jw_sum_global, 1,
               sizeof(real_t) == 4 ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    // ---------------------------------------------------------------
    // INTEGRATION VERIFICATION (scatter path): set known u_s / φ̈ and
    // confirm ApplySolidToFluidRHS / ApplyFluidToSolidRHS produce the
    // exact analytical surface integral.
    //
    //   Test A: u_s = (0, 0, 1) uniform. Expected fluid_rhs contribution:
    //      ∫_Γ (u_s · n̂) ψ_i dS = ∫_Γ 1 · ψ_i dS
    //   Sum over all interface DOFs: Σ_i fluid_rhs[i] = ∫_Γ 1 dS = face_area.
    //
    //   Test B: φ̈ = 1 uniform. Expected solid_rhs_d contribution:
    //      -∫_Γ φ̈ n_d ψ_i dS = -∫_Γ n_d ψ_i dS
    //   For flat interface n̂=(0,0,1): sum over solid interface DOFs in
    //   component d=z should equal -face_area; d=x,y should be exactly 0.
    // ---------------------------------------------------------------
    real_t rhs_f_sum_global = 0.0;
    real_t rhs_s_sum_global[3] = {0, 0, 0};
    {
        // Test A
        ParGridFunction u_s_test(&solid_fes);
        u_s_test = 0.0;
        {
            VectorFunctionCoefficient uz_coef(Dim, [](const Vector& x, Vector& v) {
                v = 0.0;
                v[v.Size() - 1] = 1.0;   // u_z = 1
            });
            u_s_test.ProjectCoefficient(uz_coef);
        }
        Vector fluid_rhs(fluid_fes.GetVSize());
        fluid_rhs = 0.0;
        iface.ApplySolidToFluidRHS(u_s_test, fluid_rhs);
        real_t rhs_f_sum_local = 0.0;
        for (int i = 0; i < fluid_rhs.Size(); ++i) rhs_f_sum_local += fluid_rhs[i];
        MPI_Reduce(&rhs_f_sum_local, &rhs_f_sum_global, 1,
                   sizeof(real_t) == 4 ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);

        // Test B
        ParGridFunction phi_tt_test(&fluid_fes);
        phi_tt_test = 1.0;
        Vector solid_rhs(solid_fes.GetVSize());
        solid_rhs = 0.0;
        iface.ApplyFluidToSolidRHS(phi_tt_test, solid_rhs);
        const int nscalar_s = solid_fes.GetNDofs();
        real_t rhs_s_sum_local[3] = {0, 0, 0};
        for (int d = 0; d < Dim; ++d) {
            for (int i = 0; i < nscalar_s; ++i) {
                // byNODES: vdof = scalar + d * nscalar
                rhs_s_sum_local[d] += solid_rhs[i + d * nscalar_s];
            }
        }
        MPI_Reduce(rhs_s_sum_local, rhs_s_sum_global, Dim,
                   sizeof(real_t) == 4 ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM, 0,
                   MPI_COMM_WORLD);
    }

    if (Mpi::Root()) {
        std::cout.precision(12);
        std::cout << "DIM=" << Dim << "\n";
        std::cout << "INTERFACE_ATTR=" << iface.InterfaceAttribute() << "\n";
        std::cout << "NUM_LOCAL_QUAD=" << nq_local << "\n";
        std::cout << "NUM_GLOBAL_QUAD=" << nq_total << "\n";
        std::cout << "NUM_REMOTE_FLUID_OWNED_QUAD=" << nq_remote_fl_tot << "\n";
        std::cout << "NUM_REMOTE_SOLID_OWNED_QUAD=" << nq_remote_sl_tot << "\n";
        std::cout << "SUM_N_X=" << (Dim >= 1 ? nsum_global[0] : 0.0) << "\n";
        std::cout << "SUM_N_Y=" << (Dim >= 2 ? nsum_global[1] : 0.0) << "\n";
        std::cout << "SUM_N_Z=" << (Dim >= 3 ? nsum_global[2] : 0.0) << "\n";
        std::cout << "MAX_UNIT_DEV=" << max_dev_global << "\n";
        std::cout << "EXTRACT_UN_MAX_ERR=" << max_err_global << "\n";
        std::cout << "EXTRACT_UN_MAX_ABS=" << max_abs_un_global << "\n";
        std::cout << "JACOBIAN_W_SUM=" << jw_sum_global << "\n";
        std::cout << "RHS_FLUID_SUM_u_s_z1=" << rhs_f_sum_global
                  << " (expected face_area)\n";
        std::cout << "RHS_SOLID_SUM_phi_tt1_x=" << rhs_s_sum_global[0]
                  << " (expected 0 for flat z-normal iface)\n";
        std::cout << "RHS_SOLID_SUM_phi_tt1_y=" << (Dim >= 2 ? rhs_s_sum_global[1] : 0.0)
                  << " (expected 0 for flat z-normal iface)\n";
        std::cout << "RHS_SOLID_SUM_phi_tt1_z=" << (Dim >= 3 ? rhs_s_sum_global[2] : 0.0)
                  << " (expected -face_area for flat z-normal iface)\n";
        std::cout << "PROBE_OK=1\n";
    }
    return 0;
}

int main(int argc, char* argv[])
{
    Mpi::Init(argc, argv);
    Hypre::Init();

    std::string mesh_file;
    int order = 4;
    int dim = 3;
    int fluid_attr = 1;
    int solid_attr = 2;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--mesh"        && i + 1 < argc) mesh_file  = argv[++i];
        else if (a == "--order"       && i + 1 < argc) order      = std::stoi(argv[++i]);
        else if (a == "--dim"         && i + 1 < argc) dim        = std::stoi(argv[++i]);
        else if (a == "--fluid-attr"  && i + 1 < argc) fluid_attr = std::stoi(argv[++i]);
        else if (a == "--solid-attr"  && i + 1 < argc) solid_attr = std::stoi(argv[++i]);
    }
    if (mesh_file.empty()) {
        if (Mpi::Root()) {
            std::cerr << "Usage: coupling_probe_interface --mesh FILE.msh "
                         "[--order 4] [--dim 2|3]\n";
        }
        return 1;
    }

    if (dim == 2) return RunProbe<2>(mesh_file, order, fluid_attr, solid_attr);
    if (dim == 3) return RunProbe<3>(mesh_file, order, fluid_attr, solid_attr);
    return 1;
}
