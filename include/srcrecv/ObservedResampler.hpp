/**
 * @file ObservedResampler.hpp
 * @brief Optional time-axis resampler for observed data (Lanczos/linear).
 *
 * Stdlib-only. Operates on a single 1-D trace at a time. No MPI.
 */

#ifndef SEM_OBSERVED_RESAMPLER_HPP
#define SEM_OBSERVED_RESAMPLER_HPP

#include <mfem.hpp>
#include <string>
#include <vector>

namespace SEM {

using mfem::real_t;

class ObservedResampler {
public:
    enum class Method { Linear, Lanczos };

    static Method ParseMethod(const std::string& s);

    /**
     * @brief Resample src[n_src] sampled at (src_dt, src_t0) onto dst[n_dst]
     *        sampled at (dst_dt, dst_t0).
     *
     * @param lanczos_a   Half-width (taps) for the Lanczos kernel (unused for linear).
     *
     * If src/dst grids are identical (same n, dt, t0 within 1e-12), a bit-exact
     * copy is written.
     *
     * Out-of-range taps are treated as zero (edge-zero padding).
     */
    static void Resample(const real_t* src, int n_src, real_t src_dt,
                         real_t src_t0,
                         real_t* dst, int n_dst, real_t dst_dt,
                         real_t dst_t0,
                         Method method, int lanczos_a);
};

}  // namespace SEM

#endif  // SEM_OBSERVED_RESAMPLER_HPP
