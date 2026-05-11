/**
 * @file GpuMacros.hpp
 * @brief GPU-specific macros for SEMSWS kernels
 *
 * This file provides macros for GPU optimization that differ from MFEM's approach.
 * MFEM's MFEM_UNROLL promotes loop unrolling on GPU for performance.
 * SEMSWS provides flexible unroll control for different optimization goals.
 */

#ifndef SEM_COMMON_GPU_MACROS_HPP
#define SEM_COMMON_GPU_MACROS_HPP

// =============================================================================
// SEM_NOUNROLL: Suppress loop unrolling on GPU (debug/profiling)
// =============================================================================
//
// Purpose: Reduce GPU register pressure for profiling and analysis
//
// Strategy:
//   - GPU (CUDA/HIP): Use #pragma unroll 1 to prevent loop unrolling
//   - CPU: No pragma, let compiler optimize
//
// Usage:
//   SEM_NOUNROLL
//   for (int k = 0; k < NGLL; k++) { ... }
//

#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
    #define SEM_NOUNROLL _Pragma("unroll 1")
#elif defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)
    #define SEM_NOUNROLL _Pragma("unroll 1")
#else
    #define SEM_NOUNROLL
#endif

// =============================================================================
// SEM_UNROLL5: Partial unroll (5 iterations) for production
// =============================================================================
//
// Purpose: Balance ILP (Instruction Level Parallelism) and register pressure
//
// Strategy:
//   - GPU (CUDA/HIP): Use #pragma unroll 5 for partial unrolling
//   - CPU: No pragma, let compiler optimize
//
// Benefits:
//   - NGLL <= 5: Full unroll (same as no pragma)
//   - NGLL > 5: Partial unroll, reduces register pressure while maintaining ILP
//
// Usage:
//   SEM_UNROLL5
//   for (int k = 0; k < NGLL; k++) { ... }
//

#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
    #define SEM_UNROLL5 _Pragma("unroll 5")
#elif defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)
    #define SEM_UNROLL5 _Pragma("unroll 5")
#else
    #define SEM_UNROLL5
#endif

// =============================================================================
// SEM_UNROLL3: Partial unroll (3 iterations) for lower register pressure
// =============================================================================
//
// Purpose: Further reduce register pressure while maintaining some ILP
//
// Strategy:
//   - GPU (CUDA/HIP): Use #pragma unroll 3 for minimal unrolling
//   - CPU: No pragma, let compiler optimize
//
// Benefits:
//   - Lower register usage than SEM_UNROLL5
//   - Better occupancy for register-heavy kernels
//
// Usage:
//   SEM_UNROLL3
//   for (int k = 0; k < NGLL; k++) { ... }
//

#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
    #define SEM_UNROLL3 _Pragma("unroll 3")
#elif defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)
    #define SEM_UNROLL3 _Pragma("unroll 3")
#else
    #define SEM_UNROLL3
#endif

// =============================================================================
// SEM_UNROLL2: Minimal unroll (2 iterations) for lowest register pressure
// =============================================================================
//
// Purpose: Minimize register pressure for very heavy kernels
//
// Strategy:
//   - GPU (CUDA/HIP): Use #pragma unroll 2 for minimal unrolling
//   - CPU: No pragma, let compiler optimize
//
// Benefits:
//   - Lowest register usage among partial unroll options
//   - Best occupancy for extremely register-heavy kernels
//   - Some ILP still preserved (2 iterations in flight)
//
// Usage:
//   SEM_UNROLL2
//   for (int k = 0; k < NGLL; k++) { ... }
//

#if defined(MFEM_USE_CUDA) && defined(__CUDA_ARCH__)
    #define SEM_UNROLL2 _Pragma("unroll 2")
#elif defined(MFEM_USE_HIP) && defined(__HIP_DEVICE_COMPILE__)
    #define SEM_UNROLL2 _Pragma("unroll 2")
#else
    #define SEM_UNROLL2
#endif

// =============================================================================
// GPU Shared Memory Limits
// =============================================================================
//
// CUDA/HIP GPU shared memory is limited to 48KB per thread block.
// Limits differ for single precision (4 bytes) vs double precision (8 bytes).
//
// 3D Elastic Kernel: (12 × NGLL³ + 2 × NGLL²) × sizeof(real_t)
//   Single precision (MFEM_USE_SINGLE):
//     - NGLL=9:  35.6 KB (OK)
//     - NGLL=10: 48.5 KB (exceeds limit)
//   Double precision (default):
//     - NGLL=7:  33.7 KB (OK)
//     - NGLL=8:  50.2 KB (exceeds limit)
//
// 2D Elastic Kernel: 8 × NGLL² × sizeof(real_t)
//   Single precision (MFEM_USE_SINGLE):
//     - NGLL=38: 46.2 KB (OK)
//     - NGLL=39: 48.7 KB (exceeds limit)
//   Double precision (default):
//     - NGLL=27: 46.7 KB (OK)
//     - NGLL=28: 50.2 KB (exceeds limit)
//
// See SEMKernelDispatch.hpp for conditional instantiation based on these limits.
//

namespace SEM {

// Maximum NGLL for GPU kernels (shared memory constraint)
#ifdef MFEM_USE_SINGLE
    // Single precision (4 bytes per real_t)
    constexpr int MAX_NGLL_GPU_2D = 38;
    constexpr int MAX_NGLL_GPU_3D = 9;
#else
    // Double precision (8 bytes per real_t)
    constexpr int MAX_NGLL_GPU_2D = 27;
    constexpr int MAX_NGLL_GPU_3D = 7;
#endif

}  // namespace SEM

#endif  // SEM_COMMON_GPU_MACROS_HPP
