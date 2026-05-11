/**
 * @file ObservedResampler.cpp
 */

#include "srcrecv/ObservedResampler.hpp"

#include <mfem.hpp>
#include <cmath>
#include <cstring>

namespace SEM {

namespace {
constexpr double kPi = 3.14159265358979323846;

inline double SincNorm(double x) {
    if (std::abs(x) < 1e-12) return 1.0;
    double a = kPi * x;
    return std::sin(a) / a;
}

inline double Lanczos(double x, int a) {
    if (std::abs(x) >= a) return 0.0;
    return SincNorm(x) * SincNorm(x / static_cast<double>(a));
}
}  // namespace

ObservedResampler::Method
ObservedResampler::ParseMethod(const std::string& s) {
    if (s == "lanczos")   return Method::Lanczos;
    if (s == "linear") return Method::Linear;
    MFEM_ABORT("ObservedResampler: unknown method '" << s
               << "' (valid: lanczos, linear)");
    return Method::Linear;
}

void ObservedResampler::Resample(const real_t* src, int n_src, real_t src_dt,
                                 real_t src_t0,
                                 real_t* dst, int n_dst, real_t dst_dt,
                                 real_t dst_t0,
                                 Method method, int lanczos_a) {
    MFEM_VERIFY(src && dst, "Resample: null buffer");
    MFEM_VERIFY(n_src > 0 && n_dst > 0, "Resample: non-positive size");
    MFEM_VERIFY(src_dt > 0.0 && dst_dt > 0.0, "Resample: non-positive dt");
    if (method == Method::Lanczos) {
        MFEM_VERIFY(lanczos_a > 0, "Resample: lanczos_a must be >= 1");
    }

    // Identity fast-path (bit-exact).
    const double tol = 1e-12;
    if (n_src == n_dst &&
        std::abs(static_cast<double>(src_dt) - dst_dt) <= tol &&
        std::abs(static_cast<double>(src_t0) - dst_t0) <= tol) {
        std::memcpy(dst, src, sizeof(real_t) * static_cast<size_t>(n_src));
        return;
    }

    // Anti-alias scaling when downsampling (stretch the kernel).
    const double ratio = static_cast<double>(dst_dt) / static_cast<double>(src_dt);
    const double kernel_scale = (ratio > 1.0) ? ratio : 1.0;

    for (int i = 0; i < n_dst; ++i) {
        const double t = static_cast<double>(dst_t0)
                       + static_cast<double>(dst_dt) * i;
        const double u = (t - static_cast<double>(src_t0))
                       / static_cast<double>(src_dt);  // src-grid coord

        if (method == Method::Linear) {
            const int j0 = static_cast<int>(std::floor(u));
            const double frac = u - j0;
            const double a = (j0 >= 0 && j0 < n_src)
                             ? static_cast<double>(src[j0]) : 0.0;
            const double b = (j0 + 1 >= 0 && j0 + 1 < n_src)
                             ? static_cast<double>(src[j0 + 1]) : 0.0;
            dst[i] = static_cast<real_t>((1.0 - frac) * a + frac * b);
        } else {
            const int center = static_cast<int>(std::floor(u));
            const int half = static_cast<int>(
                std::ceil(lanczos_a * kernel_scale));
            double acc = 0.0;
            double wsum = 0.0;
            for (int k = center - half + 1; k <= center + half; ++k) {
                if (k < 0 || k >= n_src) continue;
                const double x = (u - k) / kernel_scale;
                const double w = Lanczos(x, lanczos_a);
                acc += w * static_cast<double>(src[k]);
                wsum += w;
            }
            // Normalize to preserve DC and reduce edge bias.
            if (std::abs(wsum) > 1e-12) acc /= wsum;
            dst[i] = static_cast<real_t>(acc);
        }
    }
}

}  // namespace SEM
