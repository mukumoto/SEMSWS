// observed_resampler_test — Lanczos / linear resampling accuracy.

#include "srcrecv/ObservedResampler.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using SEM::ObservedResampler;
using SEM::real_t;

#define REQUIRE(expr) do { \
    if (!(expr)) { \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expr); \
        std::exit(1); \
    } \
} while (0)

namespace {

double RMSError(const std::vector<real_t>& a, const std::vector<real_t>& b,
                int skip_edge) {
    const int n = static_cast<int>(a.size());
    double acc = 0.0;
    int cnt = 0;
    for (int i = skip_edge; i < n - skip_edge; ++i) {
        double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        acc += d * d;
        ++cnt;
    }
    return std::sqrt(acc / cnt);
}

void TestSinusoidLanczos() {
    const double f = 5.0;       // Hz
    const double src_dt = 1e-3; // f_nyq = 500 Hz; f/f_nyq = 0.01 (well under 0.4)
    const int n_src = 1024;
    const double src_t0 = 0.0;

    std::vector<real_t> src(n_src);
    for (int i = 0; i < n_src; ++i) {
        double t = src_t0 + src_dt * i;
        src[i] = static_cast<real_t>(std::sin(2.0 * M_PI * f * t));
    }

    const double dst_dt = src_dt * 0.5;
    const int n_dst = n_src * 2;
    std::vector<real_t> dst(n_dst), analytic(n_dst);
    for (int i = 0; i < n_dst; ++i) {
        double t = src_t0 + dst_dt * i;
        analytic[i] = static_cast<real_t>(std::sin(2.0 * M_PI * f * t));
    }

    ObservedResampler::Resample(src.data(), n_src, src_dt, src_t0,
                                dst.data(), n_dst, dst_dt, src_t0,
                                ObservedResampler::Method::Lanczos, 8);
    double err = RMSError(dst, analytic, 32);
    std::fprintf(stderr, "sinc up-2x RMS = %g\n", err);
    REQUIRE(err < 1e-3);

    // Down-sample 3x
    const double dst_dt3 = src_dt * 3.0;
    const int n_dst3 = n_src / 3;
    std::vector<real_t> dst3(n_dst3), analytic3(n_dst3);
    for (int i = 0; i < n_dst3; ++i) {
        double t = src_t0 + dst_dt3 * i;
        analytic3[i] = static_cast<real_t>(std::sin(2.0 * M_PI * f * t));
    }
    ObservedResampler::Resample(src.data(), n_src, src_dt, src_t0,
                                dst3.data(), n_dst3, dst_dt3, src_t0,
                                ObservedResampler::Method::Lanczos, 8);
    double err3 = RMSError(dst3, analytic3, 16);
    std::fprintf(stderr, "sinc down-3x RMS = %g\n", err3);
    REQUIRE(err3 < 1e-3);
}

void TestSinusoidLinear() {
    const double f = 5.0;
    const double src_dt = 1e-3;
    const int n_src = 1024;
    const double src_t0 = 0.0;
    std::vector<real_t> src(n_src);
    for (int i = 0; i < n_src; ++i) {
        double t = src_t0 + src_dt * i;
        src[i] = static_cast<real_t>(std::sin(2.0 * M_PI * f * t));
    }

    const double dst_dt = src_dt * 0.5;
    const int n_dst = n_src * 2;
    std::vector<real_t> dst(n_dst), analytic(n_dst);
    for (int i = 0; i < n_dst; ++i) {
        double t = src_t0 + dst_dt * i;
        analytic[i] = static_cast<real_t>(std::sin(2.0 * M_PI * f * t));
    }
    ObservedResampler::Resample(src.data(), n_src, src_dt, src_t0,
                                dst.data(), n_dst, dst_dt, src_t0,
                                ObservedResampler::Method::Linear, 0);
    double err = RMSError(dst, analytic, 4);
    std::fprintf(stderr, "linear up-2x RMS = %g\n", err);
    // f*dt_src = 0.005 so linear is plenty accurate here.
    REQUIRE(err < 2e-2);
}

void TestIdentity() {
    const int n = 32;
    const double dt = 2e-3, t0 = 0.1;
    std::vector<real_t> src(n), dst(n);
    for (int i = 0; i < n; ++i) src[i] = static_cast<real_t>(i * 0.13 - 0.5);

    ObservedResampler::Resample(src.data(), n, dt, t0,
                                dst.data(), n, dt, t0,
                                ObservedResampler::Method::Lanczos, 8);
    for (int i = 0; i < n; ++i) REQUIRE(dst[i] == src[i]);

    std::vector<real_t> dst2(n);
    ObservedResampler::Resample(src.data(), n, dt, t0,
                                dst2.data(), n, dt, t0,
                                ObservedResampler::Method::Linear, 0);
    for (int i = 0; i < n; ++i) REQUIRE(dst2[i] == src[i]);
}

void TestParseMethod() {
    REQUIRE(ObservedResampler::ParseMethod("lanczos")   == ObservedResampler::Method::Lanczos);
    REQUIRE(ObservedResampler::ParseMethod("linear") == ObservedResampler::Method::Linear);
}

}  // namespace

int main() {
    TestParseMethod();
    TestIdentity();
    TestSinusoidLanczos();
    TestSinusoidLinear();
    return 0;
}
