// acoustic_ad_vs_hand_3d_test — 3D counterpart of acoustic_ad_vs_hand_test.
//
// Same fixture style: small Cartesian hex mesh, synthetic material +
// wavefield, N accumulation steps, then assert L2 relative error between
// hand and AD kernels < 1e-6.

#include "fwi/IsotropicAcousticSensitivity3D.hpp"
#include "fwi/IsotropicAcousticSensitivityAD3D.hpp"
#include "material/MaterialField.hpp"

#include <mfem.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>

using mfem::real_t;
using mfem::Vector;
using mfem::Mesh;
using mfem::ParMesh;
using mfem::H1_FECollection;
using mfem::ParFiniteElementSpace;

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        std::exit(1); \
    } \
} while (0)

namespace {

real_t L2RelErr(const Vector& a, const Vector& b) {
    const real_t* ha = a.HostRead();
    const real_t* hb = b.HostRead();
    const int n = a.Size();
    double sum_diff_sq = 0.0, sum_ref_sq = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = static_cast<double>(ha[i]) - static_cast<double>(hb[i]);
        sum_diff_sq += d * d;
        sum_ref_sq  += static_cast<double>(hb[i]) * static_cast<double>(hb[i]);
    }
    if (sum_ref_sq == 0.0) return std::sqrt(sum_diff_sq);
    return static_cast<real_t>(std::sqrt(sum_diff_sq / sum_ref_sq));
}

std::unique_ptr<ParMesh> MakeUnitCubeHexMesh(int nx, int ny, int nz) {
    Mesh serial = Mesh::MakeCartesian3D(nx, ny, nz,
                                        mfem::Element::HEXAHEDRON);
    return std::make_unique<ParMesh>(MPI_COMM_WORLD, serial);
}

void FillRandom(Vector& v, uint64_t seed, real_t lo, real_t hi) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<real_t> dist(lo, hi);
    real_t* h = v.HostWrite();
    for (int i = 0; i < v.Size(); i++) h[i] = dist(rng);
}

void FillRandomMaterial3D(SEM::MaterialField3D& f, uint64_t seed,
                          real_t lo, real_t hi) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<real_t> dist(lo, hi);
    real_t* h = f.HostWrite();
    for (int i = 0; i < f.Size(); i++) h[i] = dist(rng);
}

void RunOneCase(int order, int nx, int ny, int nz, int nsteps, real_t tol) {
    const int ngll = order + 1;
    auto pmesh = MakeUnitCubeHexMesh(nx, ny, nz);
    H1_FECollection fec(order, /*dim=*/3);
    ParFiniteElementSpace fes(pmesh.get(), &fec);

    const int ne = pmesh->GetNE();
    SEM::MaterialField3D kappa(ne, ngll, ngll, ngll);
    SEM::MaterialField3D inv_rho(ne, ngll, ngll, ngll);

    // Physically reasonable: κ in [1e9, 5e9], ρ in [1000, 3000].
    FillRandomMaterial3D(kappa,  /*seed*/ 0xA5A5A5 ^ (uint64_t)order, 1.0e9, 5.0e9);
    {
        std::mt19937_64 rng(0x5A5A5A ^ (uint64_t)order);
        std::uniform_real_distribution<real_t> dist(1000.0, 3000.0);
        real_t* h = inv_rho.HostWrite();
        for (int i = 0; i < inv_rho.Size(); i++) h[i] = 1.0 / dist(rng);
    }

    SEM::IsotropicAcousticSensitivity3D   hand(fes, kappa, inv_rho);
    SEM::IsotropicAcousticSensitivityAD3D ad  (fes, kappa, inv_rho);

    const int ndof = fes.GetVSize();
    Vector fwd_p(ndof), fwd_a(ndof), adj_p(ndof);

    const real_t dt = 1.0e-3;
    for (int step = 0; step < nsteps; step++) {
        FillRandom(fwd_p, 0x1111 + step * 31 + order, 0.5, 1.5);
        FillRandom(fwd_a, 0x2222 + step * 31 + order, 2.0, 8.0);
        FillRandom(adj_p, 0x3333 + step * 31 + order, 0.5, 1.5);

        hand.Accumulate(fwd_p, fwd_a, adj_p, dt);
        ad  .Accumulate(fwd_p, fwd_a, adj_p, dt);
    }

    const real_t e_vp  = L2RelErr(ad.VpKernel(),  hand.VpKernel());
    const real_t e_rho = L2RelErr(ad.RhoKernel(), hand.RhoKernel());
    std::fprintf(stderr,
        "[3D order=%d, ne=%d, nsteps=%d] L2 rel err: Vp=%.3e, rho=%.3e  tol=%.1e\n",
        order, ne, nsteps,
        static_cast<double>(e_vp),
        static_cast<double>(e_rho),
        static_cast<double>(tol));
    REQUIRE(e_vp  < tol);
    REQUIRE(e_rho < tol);
}

}  // namespace

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    mfem::Device device("cpu");

    // Small NGLL choices keep the fixture fast: NGLL=3 (order 2) and NGLL=5
    // (order 4). 2×2×2 mesh keeps total DOF modest while exercising element-
    // boundary gathers.
    constexpr real_t tol = 1e-6;
    RunOneCase(/*order=*/2, /*nx=*/2, /*ny=*/2, /*nz=*/2, /*nsteps=*/10, tol);
    RunOneCase(/*order=*/4, /*nx=*/2, /*ny=*/2, /*nz=*/2, /*nsteps=*/5,  tol);

    if (mfem::Mpi::Root()) {
        std::fprintf(stderr, "acoustic_ad_vs_hand_3d_test: OK\n");
    }
    mfem::Mpi::Finalize();
    return 0;
}
