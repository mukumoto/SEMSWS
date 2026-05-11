// adjoint_source_operator_test — Phase A of elastic 2D FWI plan.
//
// Verifies `AdjointSource::ApplyTimeOperator` against analytic derivatives,
// and cross-checks the full (δJ/δm, δm/δu) chain via a finite-difference
// gradient test for all 8 combinations of (misfit_type ∈ {L2, NC}) × (receiver
// type ∈ {DISP, VEL, ACC, PS}).
//
// The test is pure CPU math on pre-built traces — it does not run any wave
// simulation. Intentional: we want to assert *operator correctness* in
// isolation so a sim-level regression later is attributable.

#include "srcrecv/AdjointSource.hpp"
#include "common/Types.hpp"

#include <mfem.hpp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

using mfem::real_t;
using SEM::AdjointSource;
using SEM::ReceiverType;

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        std::exit(1); \
    } \
} while (0)

// Mixed rel/abs: pass when either |a-b| < rel·max(|a|,|b|) or |a-b| < abs_tol.
#define REQUIRE_NEAR_MIX(a, b, rel_tol, abs_tol) do { \
    const real_t _A = static_cast<real_t>(a); \
    const real_t _B = static_cast<real_t>(b); \
    const real_t _den = std::max(std::abs(_A), std::abs(_B)); \
    const real_t _abs = std::abs(_A - _B); \
    if (_abs > (abs_tol) && _abs > (rel_tol) * _den) { \
        std::fprintf(stderr, \
            "FAIL %s:%d: |%.17g - %.17g| = %.3e (rel_tol=%.2e, abs_tol=%.2e)\n", \
            __FILE__, __LINE__, _A, _B, _abs, (rel_tol), (abs_tol)); \
        std::exit(1); \
    } \
} while (0)
#define REQUIRE_NEAR_REL(a, b, tol) REQUIRE_NEAR_MIX(a, b, tol, 1e-6)

// ----------------------------------------------------------------------------
// Part 1 — Analytic verification of ApplyTimeOperator
// ----------------------------------------------------------------------------

static void test_identity_for_disp() {
    const int n = 1024;
    const real_t dt = 1e-3;
    std::vector<real_t> x(n);
    for (int i = 0; i < n; ++i) x[i] = std::sin(0.01 * i);
    std::vector<real_t> y = x;
    AdjointSource::ApplyTimeOperator(y, dt, ReceiverType::Displacement);
    for (int i = 0; i < n; ++i) REQUIRE(y[i] == x[i]);
    std::puts("  [ok] DISP: identity");
}

static void test_first_derivative_for_vel() {
    // Cubic polynomial: central 2nd-order stencil is exact on quadratics, so
    // we pick x(t) = t²/2 + t — exact analytic result, no truncation noise.
    // Then -d/dt x(t) = -(t + 1).
    const int n = 1001;
    const real_t dt = 1e-2;
    std::vector<real_t> x(n), ref(n);
    for (int i = 0; i < n; ++i) {
        real_t t = i * dt;
        x[i]   = 0.5 * t * t + t;
        ref[i] = -(t + 1.0);
    }
    std::vector<real_t> y = x;
    AdjointSource::ApplyTimeOperator(y, dt, ReceiverType::Velocity);
    real_t max_err = 0, max_ref = 0;
    const int pad = 5;
    for (int i = pad; i < n - pad; ++i) {
        max_err = std::max(max_err, std::abs(y[i] - ref[i]));
        max_ref = std::max(max_ref, std::abs(ref[i]));
    }
    REQUIRE(max_err / max_ref < 1e-4);
    std::puts("  [ok] VEL: -d/dt matches analytic on quadratic");
}

static void test_second_derivative_for_acc_and_ps() {
    // Quadratic: central 2nd-diff is exact up to cubic, so x(t) = t²/2 + t
    // yields d²/dt² x = 1. With single-precision real_t, cancellation noise
    // from computing (x[t+1] - 2x[t] + x[t-1]) / dt² is ~ε·x_max / dt², so we
    // keep t-range and dt balanced (small t-range, coarse-ish dt).
    const int n = 101;
    const real_t dt = 1e-2;
    std::vector<real_t> x(n), ref(n);
    for (int i = 0; i < n; ++i) {
        real_t t = i * dt;
        x[i]   = 0.5 * t * t + t;
        ref[i] = 1.0;
    }
    for (ReceiverType t : {ReceiverType::Acceleration, ReceiverType::Pressure}) {
        std::vector<real_t> y = x;
        AdjointSource::ApplyTimeOperator(y, dt, t);
        real_t max_err = 0;
        const int pad = 2;
        for (int i = pad; i < n - pad; ++i) {
            max_err = std::max(max_err, std::abs(y[i] - ref[i]));
        }
        // Float roundoff budget for this stencil scales as ε·x_max/dt² ~ 2e-3.
        REQUIRE(max_err < 1e-2);
    }
    std::puts("  [ok] ACC / PS: +d²/dt² matches analytic on quadratic");
}

// ----------------------------------------------------------------------------
// Part 2 — FD gradient check: (δJ/δs) · Δ ≈ (J(s+εΔ) − J(s−εΔ))/(2ε)
//
// We reconstruct the math locally instead of wiring up a fake ReceiverArray:
//   1. Build s(t), d(t), w(t)
//   2. Compute J(s, d, w) for each misfit
//   3. Compute w_adj ≡ δJ/δm using the same formulas AdjointSource uses
//   4. Apply ApplyTimeOperator(w_adj, dt, type) → f_math ≡ δJ/δs(τ)
//   5. FD: J(s+εΔ) − J(s−εΔ) / (2ε) should equal ∫ f_math · Δ dt
// ----------------------------------------------------------------------------

// L2 waveform misfit: J = 0.5 ∫ w (s - d)² dt
static real_t misfit_L2(const std::vector<real_t>& s,
                        const std::vector<real_t>& d,
                        const std::vector<real_t>& w,
                        real_t dt) {
    real_t J = 0.0;
    const int n = static_cast<int>(s.size());
    for (int t = 0; t < n; ++t) {
        real_t r = s[t] - d[t];
        J += 0.5 * w[t] * r * r * dt;
    }
    return J;
}

// NC misfit (gradient-consistent form): J = 0.5 ∫ w (s/‖s‖ - d/‖d‖)² dt
static real_t misfit_NC(const std::vector<real_t>& s,
                        const std::vector<real_t>& d,
                        const std::vector<real_t>& w,
                        real_t dt) {
    real_t ns2 = 0, nd2 = 0;
    const int n = static_cast<int>(s.size());
    for (int t = 0; t < n; ++t) {
        ns2 += s[t] * s[t] * w[t] * dt;
        nd2 += d[t] * d[t] * w[t] * dt;
    }
    if (ns2 <= 0 || nd2 <= 0) return 0.0;
    real_t ns = std::sqrt(ns2), nd = std::sqrt(nd2);
    real_t J = 0.0;
    for (int t = 0; t < n; ++t) {
        real_t r = s[t] / ns - d[t] / nd;
        J += 0.5 * w[t] * r * r * dt;
    }
    return J;
}

// w_adj = δJ/δm for L2:   w·(s - d)
// w_adj = δJ/δm for NC:   (w/‖s‖) (s·cc/(‖s‖²‖d‖) − d/‖d‖)
static std::vector<real_t> w_adj_L2(const std::vector<real_t>& s,
                                    const std::vector<real_t>& d,
                                    const std::vector<real_t>& w) {
    std::vector<real_t> out(s.size(), 0.0);
    for (size_t t = 0; t < s.size(); ++t) out[t] = w[t] * (s[t] - d[t]);
    return out;
}
static std::vector<real_t> w_adj_NC(const std::vector<real_t>& s,
                                    const std::vector<real_t>& d,
                                    const std::vector<real_t>& w,
                                    real_t dt) {
    const int n = static_cast<int>(s.size());
    real_t ns2 = 0, nd2 = 0, cc = 0;
    for (int t = 0; t < n; ++t) {
        ns2 += s[t] * s[t] * w[t] * dt;
        nd2 += d[t] * d[t] * w[t] * dt;
        cc  += s[t] * d[t] * w[t] * dt;
    }
    std::vector<real_t> out(n, 0.0);
    if (ns2 <= 0 || nd2 <= 0) return out;
    real_t ns = std::sqrt(ns2), nd = std::sqrt(nd2);
    for (int t = 0; t < n; ++t) {
        out[t] = (w[t] / ns) * (s[t] * cc / (ns * ns * nd) - d[t] / nd);
    }
    return out;
}

// Build m = O[s] for each type so FD uses the right observed quantity.
static std::vector<real_t> apply_O(const std::vector<real_t>& s,
                                   real_t dt, ReceiverType type) {
    const int n = static_cast<int>(s.size());
    std::vector<real_t> m(n, 0.0);
    if (type == ReceiverType::Displacement) {
        m = s;
    } else if (type == ReceiverType::Velocity) {
        // d/dt via central (endpoints one-sided)
        real_t inv_2dt = 0.5 / dt;
        for (int t = 1; t < n - 1; ++t) m[t] = (s[t+1] - s[t-1]) * inv_2dt;
        m[0]     = (s[1] - s[0]) / dt;
        m[n - 1] = (s[n - 1] - s[n - 2]) / dt;
    } else {
        // ACC or PS: d²/dt² (sign for PS absorbed: we treat m directly so sign
        // cancels when both obs and syn are transformed the same way).
        real_t inv_dt2 = 1.0 / (dt * dt);
        for (int t = 1; t < n - 1; ++t) {
            m[t] = (s[t+1] - 2.0 * s[t] + s[t-1]) * inv_dt2;
        }
    }
    return m;
}

static void fd_check(const char* tag, bool use_NC, ReceiverType type) {
    const int n = 2001;
    const real_t dt = 1e-3;
    const real_t omega = 2.0 * M_PI * 2.0;  // 2 Hz — well sampled

    std::vector<real_t> s(n), d(n), w(n, 1.0);
    for (int i = 0; i < n; ++i) {
        real_t t = i * dt;
        s[i] = std::sin(omega * t);
        d[i] = std::sin(omega * (t - 0.05));  // 50 ms delay
    }

    // Analytic chain: f_math(τ) = ApplyTimeOperator(w_adj(m_s, m_d), dt, type)
    std::vector<real_t> m_s = apply_O(s, dt, type);
    std::vector<real_t> m_d = apply_O(d, dt, type);
    std::vector<real_t> wadj = use_NC ? w_adj_NC(m_s, m_d, w, dt)
                                      : w_adj_L2(m_s, m_d, w);
    AdjointSource::ApplyTimeOperator(wadj, dt, type);

    // Smooth direction Δ — smoothness keeps FD in the linear regime when
    // apply_O involves d/dt or d²/dt². Broadband content so the inner product
    // with the gradient isn't accidentally zero (which would make the rel
    // metric meaningless). Mix of frequencies covering the signal band.
    std::vector<real_t> delta(n, 0.0);
    const int pad = 100;
    for (int i = pad; i < n - pad; ++i) {
        real_t u = real_t(i - pad) / real_t(n - 1 - 2 * pad);   // 0..1
        real_t env = std::sin(M_PI * u);                         // cos-tapered bump
        delta[i] = env * (std::sin(2.0 * M_PI * 2.0 * u)         // 2 cycles
                        + 0.7 * std::cos(2.0 * M_PI * 2.0 * u)   // 2-cos
                        + 0.3 * std::sin(2.0 * M_PI * 5.0 * u));  // 5 cycles
    }

    // Analytic directional derivative: ⟨f_math, Δ⟩ dt (inner product)
    real_t dJ_anal = 0.0;
    for (int t = 0; t < n; ++t) dJ_anal += wadj[t] * delta[t] * dt;

    // FD step: large enough that (J+ − J−) isn't dominated by float roundoff
    // of J itself (~ε_float·|J| / (2h) noise, so need h ≫ ε_float).
    const real_t eps = 1e-2;
    auto J = [&](const std::vector<real_t>& ss) {
        std::vector<real_t> mm = apply_O(ss, dt, type);
        return use_NC ? misfit_NC(mm, m_d, w, dt)
                      : misfit_L2(mm, m_d, w, dt);
    };
    std::vector<real_t> sp(n), sm(n);
    for (int i = 0; i < n; ++i) {
        sp[i] = s[i] + eps * delta[i];
        sm[i] = s[i] - eps * delta[i];
    }
    real_t dJ_fd = (J(sp) - J(sm)) / (2.0 * eps);

    const real_t rel = std::abs(dJ_anal - dJ_fd)
                     / std::max(std::abs(dJ_anal), std::abs(dJ_fd));
    std::printf("  [%s / %s] dJ_anal=%.6e  dJ_fd=%.6e  rel=%.2e\n",
                tag, SEM::ReceiverTypeToString(type).c_str(),
                dJ_anal, dJ_fd, rel);
    // Tolerance accommodates single-precision roundoff in (J+−J−)/(2h)
    // plus, for ACC/PS, an extra O(h²) truncation from d²/dt² of the test
    // signal. 5% covers both comfortably without hiding bugs.
    real_t tol = (type == ReceiverType::Acceleration ||
                  type == ReceiverType::Pressure) ? 5e-2 : 2e-2;
    REQUIRE(rel < tol);
}

// ----------------------------------------------------------------------------
int main() {
    std::puts("== ApplyTimeOperator analytic checks ==");
    test_identity_for_disp();
    test_first_derivative_for_vel();
    test_second_derivative_for_acc_and_ps();

    std::puts("== Misfit-gradient FD consistency (8 cases) ==");
    for (auto type : {ReceiverType::Displacement, ReceiverType::Velocity,
                      ReceiverType::Acceleration, ReceiverType::Pressure}) {
        fd_check("L2", false, type);
        fd_check("NC", true,  type);
    }

    std::puts("all adjoint-source operator tests passed");
    return 0;
}
