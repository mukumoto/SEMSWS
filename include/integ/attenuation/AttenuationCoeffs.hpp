#ifndef SEM_ATTENUATION_COEFFS_HPP
#define SEM_ATTENUATION_COEFFS_HPP

/**
 * @file AttenuationCoeffs.hpp
 * @brief Attenuation coefficient computation for generalized Zener body
 *
 * Computes relaxation parameters (tau_epsilon, tau_sigma) for the
 * Generalized Zener Body viscoelastic model. Two methods are available:
 *
 * 1. Blanch tau method (default, use_optimization=true):
 *    Analytical closed-form using a single scalar parameter tau.
 *    Reference: Blanch, Robertsson & Symes, Geophysics 60, 176-184 (1995)
 *
 * 2. Classical linear least squares (use_optimization=false):
 *    Reference: Emmerich & Korn, Geophysics 52, 1252-1264 (1987)
 */

#include <mfem.hpp>
#include <string>
#include <vector>
#include <utility>

namespace SEM {

using mfem::real_t;

/**
 * @brief Result structure for attenuation coefficient computation
 */
struct AttenuationParams {
    std::vector<real_t> tau_epsilon;  ///< Relaxation times for strain
    std::vector<real_t> tau_sigma;    ///< Relaxation times for stress
    std::vector<real_t> phi;          ///< Phi coefficients for memory variables
    real_t error;                      ///< Approximation error

    /// Unrelaxed modulus correction factor for physical dispersion
    /// M_unrelaxed = M_relaxed * unrelaxed_correction
    /// This corrects for frequency-dependent modulus in viscoelastic media
    real_t unrelaxed_correction = 1.0;
};

/**
 * @brief Compute attenuation coefficients for a Generalized Zener body model
 *
 * Converts target Q factor to relaxation parameters using nonlinear optimization.
 * The frequency band is [f_min, f_max] and the optimization minimizes the error
 * in Q(f) approximation over this band.
 *
 * Internal formulas:
 *   tau_sigma = 1 / theta
 *   tau_epsilon = tau_sigma * (1 + N * weight)
 *   phi = (tau_epsilon - tau_sigma) / tau_epsilon
 *
 * @param N Number of relaxation mechanisms (typically 3-5)
 * @param Qref Target Q factor (quality factor)
 * @param f_min Minimum frequency in optimization band (Hz)
 * @param f_max Maximum frequency in optimization band (Hz)
 * @param use_optimization If true, use Blanch tau method; if false, use linear least squares
 * @return AttenuationParams containing tau_epsilon, tau_sigma, phi, and error
 */
AttenuationParams ComputeAttenuationCoeffs(
    int N,
    real_t Qref,
    real_t f_min,
    real_t f_max,
    bool use_optimization = true
);

/**
 * @brief Compute attenuation coefficients using reference frequency
 *
 * This is a convenience function that computes f_min = 0.1 * f0 and f_max = 10 * f0
 *
 * @param N Number of relaxation mechanisms
 * @param Qref Target Q factor
 * @param f0 Reference/central frequency (Hz)
 * @param use_optimization If true, use Blanch tau method; if false, linear least squares
 * @return AttenuationParams containing tau_epsilon, tau_sigma, phi, and error
 */
AttenuationParams ComputeAttenuationCoeffsFromF0(
    int N,
    real_t Qref,
    real_t f0,
    bool use_optimization = true
);

// ============================================================================
// Caching for Attenuation Coefficients
// ============================================================================

/**
 * @brief Cached version of ComputeAttenuationCoeffs
 *
 * Uses memoization to avoid recomputing coefficients for the same Q value.
 * Q values are rounded to 0.1 precision for cache key, which is sufficient
 * for physical accuracy and dramatically improves cache hit rate.
 *
 * @param N Number of relaxation mechanisms
 * @param Qref Target Q factor (will be rounded to 0.1 precision for key)
 * @param f_min Minimum frequency
 * @param f_max Maximum frequency
 * @param use_optimization If true, use Blanch tau method
 * @return AttenuationParams (from cache if available)
 */
AttenuationParams ComputeAttenuationCoeffsCached(
    int N,
    real_t Qref,
    real_t f_min,
    real_t f_max,
    bool use_optimization = false
);

/**
 * @brief Clear the attenuation coefficient cache
 */
void ClearAttenuationCache();

/**
 * @brief Get cache statistics
 * @return pair of (cache_hits, cache_misses)
 */
std::pair<size_t, size_t> GetAttenuationCacheStats();

// ============================================================================
// Q-approximation diagnostic output
// ============================================================================

/**
 * @brief Write Q^{-1}(f) diagnostic file for verifying constant-Q approximation
 *
 * Evaluates the Blanch tau (or E&K) Q^{-1} approximation over a wide frequency
 * range and writes a tab-separated file that users can plot.
 * Only call from rank 0.
 *
 * @param filepath Output file path (e.g., "output/q_approximation.dat")
 * @param N Number of SLS relaxation mechanisms
 * @param Qkappa Q factor for bulk modulus
 * @param Qmu Q factor for shear modulus (0 for acoustic)
 * @param f0 Reference frequency (Hz)
 * @param use_blanch If true, use Blanch tau method; if false, use E&K
 */
void WriteQApproximationFile(
    const std::string& filepath,
    int N, real_t Qkappa, real_t Qmu,
    real_t f0, bool use_blanch = true);

// ============================================================================
// Internal functions (exposed for testing)
// ============================================================================

namespace internal {

// ClassicalLinearLeastSquares has been replaced by EmmerichKornLeastSquares
// (clean MFEM-based rewrite). Old code is in #if 0 block in .cpp.

/**
 * @brief Blanch tau method for attenuation coefficient computation
 *
 * Analytical closed-form: sets log-spaced tau_sigma, computes zeta
 * via frequency-domain integrals, then tau = N * zeta / Q.
 *
 * Reference: Blanch, Robertsson & Symes, Geophysics 60, 176-184 (1995)
 *
 * @param N Number of relaxation mechanisms
 * @param Qref Target Q factor
 * @param f_min Minimum frequency of constant-Q band (Hz)
 * @param f_max Maximum frequency of constant-Q band (Hz)
 * @param[out] tau_sigma Stress relaxation times
 * @param[out] tau_epsilon Strain relaxation times
 * @return Blanch tau parameter
 */
real_t BlanchTauMethod(
    int N,
    real_t Qref,
    real_t f_min,
    real_t f_max,
    std::vector<real_t>& tau_sigma,
    std::vector<real_t>& tau_epsilon
);

/**
 * @brief Compute Q(f) approximation error for Blanch tau method
 */
real_t ComputeQErrorBlanch(
    int N,
    real_t Qref,
    real_t f_min,
    real_t f_max,
    const std::vector<real_t>& tau_sigma,
    real_t tau_param
);

} // namespace internal

} // namespace SEM

#endif // SEM_ATTENUATION_COEFFS_HPP
