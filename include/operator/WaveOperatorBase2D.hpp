/**
 * @file WaveOperatorBase2D.hpp
 * @brief Intermediate base class for 2D wave operators
 *
 * This class provides common implementation for 2D wave operators,
 * reducing code duplication between Elastic and Acoustic operators.
 */

#ifndef SEM_OPERATOR_WAVE_OPERATOR_BASE_2D_HPP
#define SEM_OPERATOR_WAVE_OPERATOR_BASE_2D_HPP

#include "operator/WaveOperator.hpp"
#include "integ/SEMIntegrators.hpp"

namespace SEM {

/**
 * @class WaveOperatorBase2D
 * @brief Intermediate base class with common 2D wave operator implementation
 *
 * Provides shared functionality for:
 * - Diagonal mass matrix assembly
 * - Damping (Shin 1995 ABC) assembly
 * - Explicit time stepping
 * - Memory management
 *
 * Derived classes only need to implement:
 * - SetupMass() - call AssembleDiagonalMass() with appropriate coefficient
 * - SetupStiffness() - create physics-specific integrators
 * - PrintInfo() - custom information output
 */
class WaveOperatorBase2D : public WaveOperator {
public:
    /**
     * @brief Construct base 2D operator
     * @param fes Finite element space
     * @param order Polynomial order
     */
    WaveOperatorBase2D(ParFiniteElementSpace& fes, int order);

    ~WaveOperatorBase2D() override;

    // -------------------------------------------------------------------------
    // WaveOperator interface - common implementations
    // -------------------------------------------------------------------------

    WaveOperator& SetupSource(PointSourceCollection& source) override;

    void ResetState() override;

    void ExplicitSolve(const ParGridFunction& u,
                       const ParGridFunction& dudt,
                       ParGridFunction& dudt2,
                       real_t dt) override;

    /**
     * @brief Set time step for viscoelastic integrator
     * @param dt Time step
     *
     * CRITICAL: Must be called BEFORE SetupStiffness() when using viscoelasticity.
     * EnableAttenuation() uses dt to compute Crank-Nicolson coefficients.
     * If dt=0, all viscoelastic effects become zero.
     */
    void SetTimeStep(real_t dt) { dt_ = dt; }

    /**
     * @brief Release setup-only resources to save memory
     *
     * Frees mass_integ_ and mass_coef_ which are only needed during setup.
     * Call after SetupMass/SetupStiffness are complete.
     * Logs the amount of memory freed.
     */
    void FreeSetupResources() override;

    /**
     * @brief Initialize device memory for all components
     *
     * Call after SetupMass/SetupStiffness are complete.
     * Enables and syncs device memory for:
     * - All domain integrators (geometry, DOFs, material params)
     * - Mass matrix (M_inv_)
     * - RHS workspace
     */
    void DeviceInit();

    // -------------------------------------------------------------------------
    // GPU kernel helper methods (must be public for CUDA extended lambda)
    // -------------------------------------------------------------------------

    /**
     * @brief Assemble domain contribution to RHS via domain_integs_
     * @param u Input displacement field
     *
     * Sums the action of all registered domain integrators into rhs_.
     * Note: Must be public for CUDA extended lambda compatibility.
     */
    void AssembleDomainRHS(const ParGridFunction& u);

    /**
     * @brief Apply inverse mass to compute acceleration
     * @param dudt Input velocity (unused when no damping)
     * @param dudt2 Output acceleration (modified in place)
     *
     * Computes: dudt2 = M_inv * rhs
     * Should be called after AssembleDomainRHS().
     * Note: Must be public for CUDA extended lambda compatibility.
     */
    void ApplyInverseMass(const ParGridFunction& dudt,
                          ParGridFunction& dudt2);

    /// See WaveOperatorBase3D::AssemblePreCouplingRHS — same semantics for 2D.
    void AssemblePreCouplingRHS(const ParGridFunction& u, real_t dt) override;

    /// See WaveOperatorBase3D::FinalizeAndApplyMass — same semantics for 2D.
    void FinalizeAndApplyMass(const ParGridFunction& dudt,
                              ParGridFunction& dudt2) override;

    /// See WaveOperatorBase3D::Rhs — same semantics for 2D.
    ParGridFunction& Rhs() { return *rhs_; }
    const ParGridFunction& Rhs() const { return *rhs_; }
    ParGridFunction*       RhsPtr() override { return rhs_.get(); }
    const ParGridFunction* RhsPtr() const override { return rhs_.get(); }

    /// See WaveOperatorBase3D::InverseMassDiagonal — same semantics for 2D.
    const ParGridFunction& InverseMassDiagonal() const { return *M_inv_; }
    const ParGridFunction* InverseMassDiagonalPtr() const override {
        return M_inv_.get();
    }


    // -------------------------------------------------------------------------
    // Mass assembly (must be public for CUDA extended lambda)
    // -------------------------------------------------------------------------

    /**
     * @brief Assemble diagonal mass matrix with given coefficient
     * @param coef Coefficient at GLL points (rho for elastic, 1/kappa for acoustic)
     * @param is_vector True for vector field (elastic), false for scalar (acoustic)
     *
     * After calling, M_inv_ contains the inverted mass matrix.
     * The coef is stored in mass_coef_ for potential damping use.
     * Note: Must be public for CUDA extended lambda compatibility.
     */
    void AssembleDiagonalMass(const Vector& coef, bool is_vector);

    /**
     * @brief Setup Cerjan multiplicative taper
     * @param config Damping configuration (alpha interpreted as abpc percentage)
     *
     * Computes taper vector and stores for per-step application.
     * Does NOT modify mass or stiffness matrices.
     */
    void SetupCerjanTaper(const DampingConfig& config);

    /**
     * @brief Apply Cerjan taper to all state vectors (GPU-portable)
     */
    void ApplyCerjanTaper(ParGridFunction& u, ParGridFunction& dudt,
                           ParGridFunction& dudt2) override;

protected:

    /**
     * @brief Hook for derived classes to apply boundary conditions
     *
     * Called during ExplicitSolve() after parallel assembly.
     * Default: no-op. Override in AcousticOperator for Dirichlet BC.
     */
    virtual void ApplyBoundaryConditions() {}

    /**
     * @brief Hook for derived classes to reset physics-specific state
     *
     * Called by ResetState(). Override for viscoelastic memory reset.
     */
    virtual void ResetPhysicsState() {}

    // -------------------------------------------------------------------------
    // Protected members accessible to derived classes
    // -------------------------------------------------------------------------

    int order_;

    // Mass integrator and matrices
    std::unique_ptr<SEMMassIntegrator2D> mass_integ_;
    std::unique_ptr<ParGridFunction> M_inv_;   // Inverse mass (diagonal)

    // RHS workspace
    std::unique_ptr<ParGridFunction> rhs_;
    Vector rhs_vec_;

    // Coefficient storage (for damping computation)
    Vector mass_coef_;

    // Time step (used by viscoelastic integrator)
    real_t dt_ = 0.0;

    // Damping configuration (stored for PrintInfo/summary)
    DampingConfig damping_config_;

    // Cerjan taper vector (same size as u, stored on device for GPU kernels)
    std::unique_ptr<ParGridFunction> cerjan_taper_;

    // Stiffness integrators (owned by this class)
    Array<BilinearFormIntegrator*> domain_integs_;

    // External source
    PointSourceCollection* source_ = nullptr;
};

}  // namespace SEM

#endif  // SEM_OPERATOR_WAVE_OPERATOR_BASE_2D_HPP
