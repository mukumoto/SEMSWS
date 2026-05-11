// elastic_ad_vs_hand_3d_test — 3D elastic hand-vs-AD bitwise-close check.

#include "fwi/IsotropicElasticSensitivity3D.hpp"
#include "fwi/IsotropicElasticSensitivityAD3D.hpp"
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
    ParFiniteElementSpace fes(pmesh.get(), &fec, /*vdim=*/3);

    const int ne = pmesh->GetNE();
    SEM::MaterialField3D lambda(ne, ngll, ngll, ngll);
    SEM::MaterialField3D mu(ne, ngll, ngll, ngll);
    SEM::MaterialField3D rho(ne, ngll, ngll, ngll);

    // Physically reasonable: λ, μ in [1e9, 5e9], ρ in [1000, 3000].
    FillRandomMaterial3D(lambda, 0xAAA1 ^ (uint64_t)order, 1.0e9, 5.0e9);
    FillRandomMaterial3D(mu,     0xAAA2 ^ (uint64_t)order, 1.0e9, 5.0e9);
    FillRandomMaterial3D(rho,    0xAAA3 ^ (uint64_t)order, 1000.0, 3000.0);

    SEM::IsotropicElasticSensitivity3D   hand(fes, lambda, mu, rho);
    SEM::IsotropicElasticSensitivityAD3D ad  (fes, lambda, mu, rho);

    const int ndof = fes.GetVSize();
    Vector fwd_u(ndof), fwd_a(ndof), adj_u(ndof);

    const real_t dt = 1.0e-3;
    for (int step = 0; step < nsteps; step++) {
        FillRandom(fwd_u, 0x1111 + step * 31 + order, -1.0, 1.0);
        FillRandom(fwd_a, 0x2222 + step * 31 + order, -2.0, 2.0);
        FillRandom(adj_u, 0x3333 + step * 31 + order, -1.0, 1.0);
        hand.Accumulate(fwd_u, fwd_a, adj_u, dt);
        ad  .Accumulate(fwd_u, fwd_a, adj_u, dt);
    }

    // Need to retrieve (λ, μ, ρ) kernels from both backends. The public
    // interface exposes them in the same-named methods (LambdaKernel,
    // MuKernel, RhoKernel) on both classes.
    const real_t e_lam = L2RelErr(ad.LambdaKernel(), hand.LambdaKernel());
    const real_t e_mu  = L2RelErr(ad.MuKernel(),     hand.MuKernel());
    const real_t e_rho = L2RelErr(ad.RhoKernel(),    hand.RhoKernel());
    std::fprintf(stderr,
        "[3D elastic order=%d, ne=%d, nsteps=%d] L2 rel err: "
        "λ=%.3e, μ=%.3e, ρ=%.3e  tol=%.1e\n",
        order, ne, nsteps,
        static_cast<double>(e_lam),
        static_cast<double>(e_mu),
        static_cast<double>(e_rho),
        static_cast<double>(tol));
    REQUIRE(e_lam < tol);
    REQUIRE(e_mu  < tol);
    REQUIRE(e_rho < tol);
}

}  // namespace

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    mfem::Device device("cpu");

    // Heavier than acoustic (vdim=3), so use fewer cases.
    constexpr real_t tol = 1e-6;
    RunOneCase(/*order=*/2, /*nx=*/2, /*ny=*/2, /*nz=*/2, /*nsteps=*/10, tol);
    RunOneCase(/*order=*/3, /*nx=*/2, /*ny=*/2, /*nz=*/2, /*nsteps=*/5,  tol);

    if (mfem::Mpi::Root()) {
        std::fprintf(stderr, "elastic_ad_vs_hand_3d_test: OK\n");
    }
    mfem::Mpi::Finalize();
    return 0;
}
