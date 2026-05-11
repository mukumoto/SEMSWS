/**
 * @file SEMKernelDispatch.hpp
 * @brief Template dispatch macros for SEM integrator kernels
 *
 * Provides macros for dispatching to NGLL-specialized template kernels.
 *
 * NGLL support matrix:
 *
 * Single precision (MFEM_USE_SINGLE):
 * - 3D kernels: NGLL 2-9  on GPU, NGLL 2-18 on CPU
 * - 2D kernels: NGLL 2-18 on GPU and CPU
 *
 * Double precision (default):
 * - 3D kernels: NGLL 2-7  on GPU, NGLL 2-18 on CPU
 * - 2D kernels: NGLL 2-18 on GPU and CPU
 *
 * NGLL outside the supported range aborts at runtime via MFEM_ABORT.
 */

#ifndef SEM_KERNEL_DISPATCH_HPP
#define SEM_KERNEL_DISPATCH_HPP

// =============================================================================
// GPU Detection
// =============================================================================

// Key on both the mfem-level and project-level GPU flags. The project flag
// (SEM_USE_HIP / SEM_USE_CUDA) is set by src/CMakeLists.txt in lock-step with
// the mfem one and is reliably visible via -D on the compile line, whereas
// MFEM_USE_HIP/CUDA depend on the installed mfem _config.hpp — on LUMI that
// header has been observed not to propagate the flag, which silently disabled
// the 3D-GPU instantiation clamp and overflowed gfx90a's 64 KB LDS limit.
#if defined(MFEM_USE_CUDA) || defined(MFEM_USE_HIP) || \
    defined(SEM_USE_CUDA)  || defined(SEM_USE_HIP)
#define SEM_USE_GPU 1
#else
#define SEM_USE_GPU 0
#endif

// =============================================================================
// 2D Dispatch Macro (NGLL 2-18, both GPU and CPU, both precisions)
// =============================================================================

/**
 * @brief Dispatch to 2D template kernel based on NGLL value
 *
 * Covers NGLL 2-18. Aborts at runtime for values outside this range.
 *
 * @param ngll   Runtime NGLL value
 * @param method Template method name (without <NGLL>)
 * @param ...    Arguments to pass to the method
 */
#define SEM_DISPATCH_NGLL(ngll, method, ...)                                    \
    switch (ngll) {                                                             \
        case 2:  method<2>(__VA_ARGS__);  break;                                \
        case 3:  method<3>(__VA_ARGS__);  break;                                \
        case 4:  method<4>(__VA_ARGS__);  break;                                \
        case 5:  method<5>(__VA_ARGS__);  break;                                \
        case 6:  method<6>(__VA_ARGS__);  break;                                \
        case 7:  method<7>(__VA_ARGS__);  break;                                \
        case 8:  method<8>(__VA_ARGS__);  break;                                \
        case 9:  method<9>(__VA_ARGS__);  break;                                \
        case 10: method<10>(__VA_ARGS__); break;                                \
        case 11: method<11>(__VA_ARGS__); break;                                \
        case 12: method<12>(__VA_ARGS__); break;                                \
        case 13: method<13>(__VA_ARGS__); break;                                \
        case 14: method<14>(__VA_ARGS__); break;                                \
        case 15: method<15>(__VA_ARGS__); break;                                \
        case 16: method<16>(__VA_ARGS__); break;                                \
        case 17: method<17>(__VA_ARGS__); break;                                \
        case 18: method<18>(__VA_ARGS__); break;                                \
        default:                                                                \
            MFEM_ABORT("NGLL=" << (ngll) << " not supported. Use NGLL 2-18.");  \
    }

/**
 * @brief 2D dispatch macro (alias for SEM_DISPATCH_NGLL)
 */
#define SEM_DISPATCH_NGLL_2D(ngll, method, ...) \
    SEM_DISPATCH_NGLL(ngll, method, __VA_ARGS__)

// =============================================================================
// 3D Dispatch Macro (GPU: precision-limited by shared memory; CPU: 2-18)
// =============================================================================

/**
 * @brief Dispatch to 3D template kernel
 *
 * GPU shared memory limits (48KB):
 * - Single precision: NGLL 2-9 (max 35.6 KB at NGLL=9)
 * - Double precision: NGLL 2-7 (max 33.7 KB at NGLL=7)
 *
 * On CPU builds: NGLL 2-18 supported.
 */
#if SEM_USE_GPU
    #ifdef MFEM_USE_SINGLE
        #define SEM_DISPATCH_NGLL_3D(ngll, method, ...)                                               \
            switch (ngll) {                                                                           \
                case 2:  method<2>(__VA_ARGS__);  break;                                              \
                case 3:  method<3>(__VA_ARGS__);  break;                                              \
                case 4:  method<4>(__VA_ARGS__);  break;                                              \
                case 5:  method<5>(__VA_ARGS__);  break;                                              \
                case 6:  method<6>(__VA_ARGS__);  break;                                              \
                case 7:  method<7>(__VA_ARGS__);  break;                                              \
                case 8:  method<8>(__VA_ARGS__);  break;                                              \
                case 9:  method<9>(__VA_ARGS__);  break;                                              \
                default:                                                                              \
                    MFEM_ABORT("NGLL=" << (ngll) << " not supported on GPU (single). Use NGLL 2-9."); \
            }
    #else
        #define SEM_DISPATCH_NGLL_3D(ngll, method, ...)                                               \
            switch (ngll) {                                                                           \
                case 2:  method<2>(__VA_ARGS__);  break;                                              \
                case 3:  method<3>(__VA_ARGS__);  break;                                              \
                case 4:  method<4>(__VA_ARGS__);  break;                                              \
                case 5:  method<5>(__VA_ARGS__);  break;                                              \
                case 6:  method<6>(__VA_ARGS__);  break;                                              \
                case 7:  method<7>(__VA_ARGS__);  break;                                              \
                default:                                                                              \
                    MFEM_ABORT("NGLL=" << (ngll) << " not supported on GPU (double). Use NGLL 2-7."); \
            }
    #endif
#else
    // CPU: NGLL 2-18 supported
    #define SEM_DISPATCH_NGLL_3D(ngll, method, ...) \
        SEM_DISPATCH_NGLL(ngll, method, __VA_ARGS__)
#endif

// =============================================================================
// Template Instantiation Macros
// =============================================================================

/**
 * @brief Instantiate templates for NGLL 2-7 (GPU 3D double precision)
 */
#define SEM_INSTANTIATE_NGLL_2_7(decl) \
    decl(2) decl(3) decl(4) decl(5) decl(6) decl(7)

/**
 * @brief Instantiate templates for NGLL 2-9 (GPU 3D single precision)
 */
#define SEM_INSTANTIATE_NGLL_2_9(decl) \
    decl(2) decl(3) decl(4) decl(5) decl(6) decl(7) decl(8) decl(9)

/**
 * @brief Instantiate templates for NGLL 2-18 (CPU, and 2D on GPU)
 */
#define SEM_INSTANTIATE_NGLL_2_18(decl)                                       \
    decl(2)  decl(3)  decl(4)  decl(5)  decl(6)  decl(7)  decl(8)  decl(9)   \
    decl(10) decl(11) decl(12) decl(13) decl(14) decl(15) decl(16) decl(17)  \
    decl(18)

/**
 * @brief 3D kernel instantiation: GPU-limited by precision
 *
 * On GPU (single): NGLL 2-9
 * On GPU (double): NGLL 2-7
 * On CPU: NGLL 2-18
 */
#if SEM_USE_GPU
    #ifdef MFEM_USE_SINGLE
        #define SEM_INSTANTIATE_NGLL_3D(decl) SEM_INSTANTIATE_NGLL_2_9(decl)
    #else
        #define SEM_INSTANTIATE_NGLL_3D(decl) SEM_INSTANTIATE_NGLL_2_7(decl)
    #endif
#else
    #define SEM_INSTANTIATE_NGLL_3D(decl) SEM_INSTANTIATE_NGLL_2_18(decl)
#endif

/**
 * @brief 2D kernel instantiation: NGLL 2-18 on both GPU and CPU
 */
#define SEM_INSTANTIATE_NGLL_2D(decl) SEM_INSTANTIATE_NGLL_2_18(decl)

#endif  // SEM_KERNEL_DISPATCH_HPP
