/**
 * @file WaveOperator.hpp
 * @brief Base class and configuration structures for wave operators
 *
 * This file provides:
 * - DampingConfig: Configuration for sponge absorbing boundary conditions
 * - WaveOperator: Base class for all wave equation operators
 */

#ifndef SEM_OPERATOR_WAVE_OPERATOR_HPP
#define SEM_OPERATOR_WAVE_OPERATOR_HPP

#include <mfem.hpp>
#include "material/Material.hpp"
#include <memory>
#include <vector>

namespace SEM {


using namespace mfem;

// Forward declarations
class PointSourceCollection;

// =============================================================================
// Configuration Structures
// =============================================================================

/**
 * @brief ABC type selector
 */
enum class ABCType {
    Cerjan   ///< Cerjan - multiplicative taper applied after each time step
};

/**
 * @struct DampingStats
 * @brief Statistics computed from damping profile
 */
struct DampingStats {
    real_t min_f = 1.0;   ///< Min damping factor f (at boundary, should be ~0.90)
    real_t max_f = 1.0;   ///< Max damping factor f (interior, should be 1.0)
    bool computed = false; ///< Whether statistics have been computed
};

/**
 * @struct DampingConfig
 * @brief Configuration for sponge absorbing boundary conditions
 *
 * Cerjan: Multiplicative taper after each time step
 *   alpha = abpc percentage (e.g., 6 = 6% decay, SeisCL-compatible)
 */
struct DampingConfig {
    ABCType type = ABCType::Cerjan;   ///< ABC type
    std::vector<real_t> abc_lengths;  ///< Sponge layer lengths for each boundary
    std::vector<real_t> alpha;        ///< Damping coefficients (interpretation depends on type)
    std::vector<int> attrs;           ///< Boundary attributes for sponge layers

    /// Statistics computed after AssembleDamping (mutable for const access)
    mutable DampingStats stats;

    DampingConfig() = default;

    DampingConfig(const std::vector<real_t>& lengths,
                  const std::vector<real_t>& a,
                  const std::vector<int>& at)
        : abc_lengths(lengths), alpha(a), attrs(at) {}

    bool IsEmpty() const { return abc_lengths.empty(); }
};


// =============================================================================
// WaveOperator Base Class
// =============================================================================

/**
 * @class WaveOperator
 * @brief Base class for wave equation operators
 *
 * This class provides the interface for second-order time-dependent operators
 * used in seismic wave simulation. It uses a builder pattern for setup.
 *
 * The wave equation: M * d2u/dt2 + C * du/dt + K * u = f
 * where M is mass, C is damping, K is stiffness, f is external force
 */
class WaveOperator : public SecondOrderTimeDependentOperator {
public:
    WaveOperator(ParFiniteElementSpace& fes);
    virtual ~WaveOperator() = default;

    // -------------------------------------------------------------------------
    // Setup methods (builder pattern for chaining)
    // -------------------------------------------------------------------------

    /// Setup mass matrix (diagonal, stored as inverse for explicit solve)
    virtual WaveOperator& SetupMass() = 0;

    /// Setup stiffness operator
    virtual WaveOperator& SetupStiffness() = 0;

    /// Setup external force source
    virtual WaveOperator& SetupSource(PointSourceCollection& source) = 0;

    /// Setup Dirichlet boundary conditions (optional)
    virtual WaveOperator& SetupDirichletBC(const Array<int>& tdof_list) {
        MFEM_WARNING("Dirichlet BC not supported for this operator type");
        return *this;
    }

    // -------------------------------------------------------------------------
    // Time integration interface
    // -------------------------------------------------------------------------

    /// Reset internal state (e.g., viscoelastic memory variables)
    virtual void ResetState() = 0;

    /// Explicit time stepping: compute dudt2 = M^{-1} * (f - K*u - C*dudt)
    virtual void ExplicitSolve(const ParGridFunction& u,
                               const ParGridFunction& dudt,
                               ParGridFunction& dudt2,
                               real_t dt) = 0;

    /// Adjoint-mode explicit time stepping.
    ///
    /// For self-adjoint operators (pure acoustic/elastic), A^T = A so the
    /// forward Step is also the correct adjoint update; the default
    /// implementation delegates to ExplicitSolve.
    ///
    /// Non-self-adjoint operators (viscoelastic with memory-variable
    /// off-diagonal coupling) must override to implement the adjoint
    /// block-transposed update. See docs for the derivation.
    virtual void AdjointExplicitSolve(const ParGridFunction& u,
                                      const ParGridFunction& dudt,
                                      ParGridFunction& dudt2,
                                      real_t dt) {
        ExplicitSolve(u, dudt, dudt2, dt);
    }

    /// Stage A of ExplicitSolve: compute rhs_ = -K*u + source (LDOF, no
    /// ParallelAssemble, no mass inversion). Concrete operators expose the
    /// resulting `rhs_` via `Rhs()` so external callers (e.g. the coupled
    /// facade) can scatter surface-integral contributions into it before
    /// invoking `FinalizeAndApplyMass`. Default implementation calls the
    /// monolithic `ExplicitSolve` as a fallback for operators that haven't
    /// been refactored yet.
    virtual void AssemblePreCouplingRHS(const ParGridFunction& u, real_t dt) {
        (void)u; (void)dt;
        MFEM_ABORT("operator does not implement AssemblePreCouplingRHS");
    }

    /// Stage B of ExplicitSolve: ParallelAssemble + boundary hook + M⁻¹·rhs_.
    /// Must be called after AssemblePreCouplingRHS (and any external
    /// contributions into `Rhs()`).
    virtual void FinalizeAndApplyMass(const ParGridFunction& dudt,
                                      ParGridFunction& dudt2) {
        (void)dudt; (void)dudt2;
        MFEM_ABORT("operator does not implement FinalizeAndApplyMass");
    }

    /// Access the internal `rhs_` ParGridFunction between the two stages.
    /// Used by the coupled facade to add fluid-solid boundary-integral
    /// contributions before ParallelAssemble + M⁻¹. Returns nullptr for
    /// operators that don't expose a raw RHS buffer.
    virtual ParGridFunction* RhsPtr() { return nullptr; }
    virtual const ParGridFunction* RhsPtr() const { return nullptr; }

    /// Enforce Dirichlet BC on state vectors (e.g., χ=χ̇=χ̈=0 for acoustic)
    virtual void EnforceDirichletBC(ParGridFunction& u,
                                    ParGridFunction& dudt,
                                    ParGridFunction& dudt2) {
        // Default: no-op (elastic doesn't need this)
    }

    // -------------------------------------------------------------------------
    // Query methods
    // -------------------------------------------------------------------------

    /// Check if Cerjan taper is enabled
    bool HasCerjanTaper() const { return has_cerjan_taper_; }

    /// Apply Cerjan multiplicative taper to all state vectors.
    /// Called by the time integrator after each complete Newmark step.
    virtual void ApplyCerjanTaper(ParGridFunction& u,
                                   ParGridFunction& dudt,
                                   ParGridFunction& dudt2) {
        // Default: no-op
    }

    /// Check if viscoelasticity is enabled
    virtual bool HasViscoelasticity() const { return false; }

    // -------------------------------------------------------------------------
    // Checkpointing: attenuation state pack/unpack
    // -------------------------------------------------------------------------

    /// Get size of attenuation state vector (0 if no attenuation)
    virtual int AttenuationStateSize() const { return 0; }

    /// Pack attenuation memory variables into a single vector
    /// @param state Output vector (must be pre-sized to AttenuationStateSize())
    virtual void GetAttenuationState(Vector& state) const {
        (void)state;
    }

    /// Restore attenuation memory variables from a packed vector
    /// @param state Input vector (size must match AttenuationStateSize())
    virtual void SetAttenuationState(const Vector& state) {
        (void)state;
    }

    /// Expose the pre-inverted diagonal mass (one value per scalar DOF for
    /// acoustic, per vector VDOF for elastic) as a non-owning pointer.
    /// Returns nullptr for operator types that don't use a lumped mass.
    /// Consumed by fluid-solid coupling to add boundary-integral
    /// contributions to the acceleration without rebuilding the Newmark
    /// integrator: `a[i] += M_inv[i] · iface_rhs[i]`.
    virtual const ParGridFunction* InverseMassDiagonalPtr() const { return nullptr; }

    /// Get current time step number
    int CurrentStep() const { return current_step_; }

    /// Print operator information
    virtual void PrintInfo(std::ostream& os = mfem::out) const;

    /**
     * @brief Release setup-only resources to save memory
     *
     * Call after SetupMass/SetupStiffness are complete.
     * Frees temporary data not needed for time stepping (e.g., mass integrator).
     */
    virtual void FreeSetupResources() {}

protected:
    ParFiniteElementSpace& fes_;
    int current_step_ = 0;
    bool has_cerjan_taper_ = false;
    bool is_setup_complete_ = false;
};

}  // namespace SEM

#endif  // SEM_OPERATOR_WAVE_OPERATOR_HPP
