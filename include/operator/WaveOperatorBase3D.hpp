/**
 * @file WaveOperatorBase3D.hpp
 * @brief Intermediate base class for 3D wave operators
 *
 * This class provides common implementation for 3D wave operators,
 * reducing code duplication between Elastic and Acoustic operators.
 */

#ifndef SEM_OPERATOR_WAVE_OPERATOR_BASE_3D_HPP
#define SEM_OPERATOR_WAVE_OPERATOR_BASE_3D_HPP

#include "operator/WaveOperator.hpp"
#include "integ/SEMIntegrators.hpp"

namespace SEM {

/**
 * @class WaveOperatorBase3D
 * @brief Intermediate base class with common 3D wave operator implementation
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
class WaveOperatorBase3D : public WaveOperator {
public:
    /**
     * @brief Construct base 3D operator
     * @param fes Finite element space
     * @param order Polynomial order
     */
    WaveOperatorBase3D(ParFiniteElementSpace& fes, int order);

    ~WaveOperatorBase3D() override;

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

    /**
     * @brief Stage A of ExplicitSolve: assemble rhs_ = -K·u + source on
     *        LDOF (no ParallelAssemble yet, no mass inversion).
     *
     * After this call, `rhs_` holds the per-rank raw right-hand side with
     * contributions from stiffness and source terms. External callers (e.g.
     * CoupledSimulationFacade) may then scatter additional contributions
     * (fluid-solid coupling surface integrals) into `rhs_` via `Rhs()`
     * before invoking `FinalizeAndApplyMass`.
     *
     * Splitting ExplicitSolve this way lets the facade reproduce the
     * "rhs += stiffness + source + coupling → ParallelAssemble →
     * M⁻¹" ordering. For non-coupled callers, `ExplicitSolve` calls
     * this + `FinalizeAndApplyMass` back-to-back, preserving the old
     * monolithic behaviour exactly.
     */
    void AssemblePreCouplingRHS(const ParGridFunction& u, real_t dt) override;

    /**
     * @brief Stage B of ExplicitSolve: ParallelAssemble + boundary hook
     *        + `dudt2 = M⁻¹ · rhs_`.
     *
     * Consumes whatever is in `rhs_` at the time of call — stiffness +
     * source + any facade-added contributions. Must be called after
     * `AssemblePreCouplingRHS` (and any external RHS contributions).
     */
    void FinalizeAndApplyMass(const ParGridFunction& dudt,
                              ParGridFunction& dudt2) override;

    /**
     * @brief Access `rhs_` between Stage A and Stage B so that external
     *        boundary-integral contributions can be scattered into it.
     *
     * Valid only between `AssemblePreCouplingRHS` and `FinalizeAndApplyMass`
     * calls. The returned ParGridFunction is also a `mfem::Vector&`, so it
     * can be passed directly to `FluidSolidInterface::Apply*RHS` overloads
     * that take `Vector&`.
     */
    ParGridFunction& Rhs() { return *rhs_; }
    const ParGridFunction& Rhs() const { return *rhs_; }
    ParGridFunction*       RhsPtr() override { return rhs_.get(); }
    const ParGridFunction* RhsPtr() const override { return rhs_.get(); }

    /**
     * @brief Access the pre-inverted diagonal mass as a ParGridFunction.
     *
     * Read-only view; used by fluid-solid coupling to add boundary-integral
     * contributions to the acceleration AFTER ExplicitSolve has written
     * `a = M^-1 · (source - K·u)`. Callers compute
     * `a[i] += M_inv[i] · iface_rhs[i]` element-wise to keep the Newmark
     * step second-order accurate at the interface.
     */
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
    std::unique_ptr<SEMMassIntegrator3D> mass_integ_;
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

#endif  // SEM_OPERATOR_WAVE_OPERATOR_BASE_3D_HPP
