// dual_smoke_test — minimal sanity check for mfem::future::dual
//
// Verifies that mfem::future::dual<real_t, real_t> is available from the
// installed MFEM, compiles cleanly, and yields correct forward-mode
// derivatives both in straight host code and inside mfem::forall.
//
// This test does NOT touch the production FWI code path — it only exercises
// the MFEM primitive that subsequent AD kernels rely on.

#include <mfem.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using mfem::real_t;
using D = mfem::future::dual<real_t, real_t>;

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        std::exit(1); \
    } \
} while (0)

#define REQUIRE_NEAR(a, b, tol) do { \
    const real_t _diff = std::abs(static_cast<real_t>(a) - static_cast<real_t>(b)); \
    if (_diff > (tol)) { \
        std::fprintf(stderr, \
            "FAIL %s:%d: |%.17g - %.17g| = %.17g > %.17g\n", \
            __FILE__, __LINE__, \
            static_cast<double>(a), static_cast<double>(b), \
            static_cast<double>(_diff), static_cast<double>(tol)); \
        std::exit(1); \
    } \
} while (0)

namespace {

// ---------------------------------------------------------------------------
// Test 1: basic scalar arithmetic
//   f(x) = x^2 + 2*x + 1, seed x=3.0, dx=1.0
//   expected: f.value = 16, f.gradient = 2*3 + 2 = 8
// ---------------------------------------------------------------------------
void TestScalarPolynomial() {
    D x{3.0, 1.0};  // value=3, seed gradient=1 (d/dx)
    D f = x * x + 2.0 * x + 1.0;
    REQUIRE_NEAR(f.value, 16.0, 0.0);     // bit-exact
    REQUIRE_NEAR(f.gradient, 8.0, 0.0);   // bit-exact
}

// ---------------------------------------------------------------------------
// Test 2: rational function
//   f(x) = 1/(x^2), seed x=2.0, dx=1.0
//   expected: f.value = 0.25, f.gradient = -2/x^3 = -0.25
// ---------------------------------------------------------------------------
void TestInverseSquare() {
    D x{2.0, 1.0};
    D f = 1.0 / (x * x);
    REQUIRE_NEAR(f.value, 0.25, 1e-15);
    REQUIRE_NEAR(f.gradient, -0.25, 1e-15);
}

// ---------------------------------------------------------------------------
// Test 3: AcousticFlux-shaped expression (closely mirrors Phase 1 goal)
//   q = inv_rho * dp_dx; seed m = inv_rho
//   Expected: q.value = inv_rho * dp_dx, q.gradient = dp_dx
//
//   Then the weak-form-factored flux:
//     flux_xi  = (qx*j0 + qy*j2)*w
//     flux_eta = (qx*j3 + qy*j1)*w
//   with qx, qy dual in inv_rho; other factors real_t.
// ---------------------------------------------------------------------------
void TestAcousticFluxShape() {
    const real_t inv_rho = 0.5;
    const real_t dp_dx = 1.7;
    const real_t dp_dy = -0.9;
    const real_t j0 = 1.1, j1 = 1.0, j2 = 0.2, j3 = -0.3;
    const real_t w = 0.4;

    D d_ir{inv_rho, 1.0};
    D qx = d_ir * dp_dx;
    D qy = d_ir * dp_dy;

    // Value part must match pure real_t computation.
    REQUIRE_NEAR(qx.value, inv_rho * dp_dx, 0.0);
    REQUIRE_NEAR(qy.value, inv_rho * dp_dy, 0.0);
    // Gradient with respect to inv_rho is simply the gradient of p.
    REQUIRE_NEAR(qx.gradient, dp_dx, 0.0);
    REQUIRE_NEAR(qy.gradient, dp_dy, 0.0);

    D flux_xi  = (qx * j0 + qy * j2) * w;
    D flux_eta = (qx * j3 + qy * j1) * w;

    // Value: same as pure real_t path.
    REQUIRE_NEAR(flux_xi.value,
                 (inv_rho * dp_dx * j0 + inv_rho * dp_dy * j2) * w, 0.0);
    REQUIRE_NEAR(flux_eta.value,
                 (inv_rho * dp_dx * j3 + inv_rho * dp_dy * j1) * w, 0.0);
    // Gradient w.r.t. inv_rho: pull inv_rho out, derivative is (dp_dx*j0+dp_dy*j2)*w etc.
    REQUIRE_NEAR(flux_xi.gradient, (dp_dx * j0 + dp_dy * j2) * w, 0.0);
    REQUIRE_NEAR(flux_eta.gradient, (dp_dx * j3 + dp_dy * j1) * w, 0.0);
}

// ---------------------------------------------------------------------------
// Test 4: dual works inside mfem::forall on CPU.
// We compute y[i] = inv_rho[i] * x[i] with inv_rho seeded; copy out
// {.value, .gradient} into two plain arrays and assert they match analytic.
// ---------------------------------------------------------------------------
void TestInForall() {
    constexpr int N = 8;
    mfem::Vector inv_rho_v(N), x_v(N), val(N), grad(N);
    for (int i = 0; i < N; ++i) {
        inv_rho_v[i] = 0.25 + 0.1 * i;
        x_v[i] = 1.0 + 0.5 * i;
    }

    const real_t* ir_r = inv_rho_v.Read();
    const real_t* x_r  = x_v.Read();
    real_t* v_w = val.Write();
    real_t* g_w = grad.Write();

    mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
        D d_ir{ir_r[i], 1.0};
        D y = d_ir * x_r[i];
        v_w[i] = y.value;
        g_w[i] = y.gradient;
    });

    // Sync-to-host (CPU device: no-op)
    val.HostRead();
    grad.HostRead();

    for (int i = 0; i < N; ++i) {
        REQUIRE_NEAR(val[i], inv_rho_v[i] * x_v[i], 0.0);
        REQUIRE_NEAR(grad[i], x_v[i], 0.0);
    }
}

}  // namespace

int main(int argc, char** argv) {
    mfem::Mpi::Init(argc, argv);
    mfem::Device device("cpu");

    TestScalarPolynomial();
    TestInverseSquare();
    TestAcousticFluxShape();
    TestInForall();

    if (mfem::Mpi::Root()) {
        std::fprintf(stderr, "dual_smoke_test: OK\n");
    }
    mfem::Mpi::Finalize();
    return 0;
}
