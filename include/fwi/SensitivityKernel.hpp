/**
 * @file SensitivityKernel.hpp
 * @brief Sensitivity kernel base class for FWI
 *
 * Accumulates sensitivity kernels from forward and adjoint wavefields.
 * Material-specific subclasses implement the actual kernel computation.
 *
 * For 2D isotropic acoustic:
 *   K_κ = -Σ_t (1/κ²) · p̈_fwd · p_adj · dt = dJ/dκ   (gradient of misfit)
 *   K_ρ = -Σ_t (1/ρ²) · ∇p_fwd · ∇p_adj · dt = dJ/dρ  (gradient of misfit)
 */

#ifndef SEM_SENSITIVITY_KERNEL_HPP
#define SEM_SENSITIVITY_KERNEL_HPP

#include <mfem.hpp>
#include <string>

namespace SEM {

using mfem::real_t;
using mfem::Vector;
using mfem::ParMesh;
using mfem::ParFiniteElementSpace;
using mfem::ParGridFunction;

/**
 * @brief Base class for 2D sensitivity kernels
 *
 * Subclasses implement material-specific kernel accumulation.
 */
class SensitivityKernelBase2D {
public:
    virtual ~SensitivityKernelBase2D() = default;

    /**
     * @brief Accumulate kernel contribution for one time step
     *
     * @param fwd_p Forward wavefield (pressure/displacement)
     * @param fwd_a Forward acceleration (p̈ for acoustic)
     * @param adj_p Adjoint wavefield
     * @param dt Time step
     */
    virtual void Accumulate(const Vector& fwd_p, const Vector& fwd_a,
                            const Vector& adj_p, real_t dt) = 0;

    /**
     * @brief Save kernels to files
     * @param dir Output directory
     * @param mesh Mesh for ParGridFunction output
     * @param source_id Source identifier
     */
    virtual void Save(const std::string& dir, ParMesh& mesh,
                      int source_id) = 0;

    /**
     * @brief Reset kernels to zero (for next source)
     */
    virtual void Reset() = 0;

    /**
     * @brief Save pseudo-Hessian diagonal to files
     * @param dir Output directory
     * @param mesh Mesh for output
     * @param source_id Source identifier
     */
    virtual void SaveHessian(const std::string& dir, ParMesh& mesh,
                             int source_id) = 0;

    /**
     * @brief Reset pseudo-Hessian to zero (for next source)
     */
    virtual void ResetHessian() = 0;

    /**
     * @brief Get number of GLL points per direction
     */
    virtual int Ngll() const = 0;
};

/**
 * @brief Base class for 3D sensitivity kernels
 *
 * Same interface as SensitivityKernelBase2D. A separate 3D base lets the
 * factory functions and AdjointSimulation distinguish at compile time
 * without conditional template machinery. No 3D kernel is implemented yet;
 * factories for 3D return nullptr or abort until a derived class exists.
 */
class SensitivityKernelBase3D {
public:
    virtual ~SensitivityKernelBase3D() = default;

    virtual void Accumulate(const Vector& fwd_p, const Vector& fwd_a,
                            const Vector& adj_p, real_t dt) = 0;

    virtual void Save(const std::string& dir, ParMesh& mesh,
                      int source_id) = 0;

    virtual void Reset() = 0;

    virtual void SaveHessian(const std::string& dir, ParMesh& mesh,
                             int source_id) = 0;

    virtual void ResetHessian() = 0;

    virtual int Ngll() const = 0;
};

}  // namespace SEM

#endif  // SEM_SENSITIVITY_KERNEL_HPP
