/**
 * @file TimeIntegrator.hpp
 * @brief Modern time integrators for wave equation
 *
 * This file provides explicit time integrators using MFEM:
 * - ExplicitTimeIntegrator: Base class
 * - NewmarkCentralDifference: Newmark-beta with beta=0, gamma=0.5
 *
 * Key features:
 * - Clean interface with Init/Step pattern
 * - Works with any WaveOperator
 * - OpenMP-parallelized vector operations
 *
 * Example usage:
 *   SEM::NewmarkCentralDifference integrator;
 *   integrator.Init(wave_op, 0.0, dt);
 *
 *   for (int n = 0; n < nt; n++) {
 *       integrator.Step(u, dudt, dudt2, t, dt);
 *   }
 */

#ifndef SEM_TIME_INTEGRATOR_HPP
#define SEM_TIME_INTEGRATOR_HPP

#include <mfem.hpp>
#include "operator/Operator.hpp"

namespace SEM {


using namespace mfem;

// =============================================================================
// ExplicitTimeIntegrator Base Class
// =============================================================================

/**
 * @class ExplicitTimeIntegrator
 * @brief Base class for explicit time integrators
 *
 * Provides interface for time stepping of second-order ODEs:
 * M * d2u/dt2 = F(t, u, du/dt)
 */
class ExplicitTimeIntegrator {
public:
    virtual ~ExplicitTimeIntegrator() = default;

    /**
     * @brief Initialize the integrator
     * @param op Wave operator to use
     * @param t0 Initial time
     * @param dt Time step
     */
    virtual void Init(WaveOperator& op, real_t t0, real_t dt) = 0;

    /**
     * @brief Advance solution by one time step
     * @param u Displacement (in/out)
     * @param dudt Velocity (in/out)
     * @param dudt2 Acceleration (out)
     * @param t Current time (updated)
     * @param dt Time step
     */
    virtual void Step(ParGridFunction& u,
                      ParGridFunction& dudt,
                      ParGridFunction& dudt2,
                      real_t& t, real_t dt) = 0;

    /// Adjoint step — identical Newmark structure but the operator's
    /// AdjointExplicitSolve is invoked in place of ExplicitSolve.
    /// Default implementation falls back to Step for operators that are
    /// self-adjoint and haven't overridden it.
    virtual void AdjointStep(ParGridFunction& u,
                             ParGridFunction& dudt,
                             ParGridFunction& dudt2,
                             real_t& t, real_t dt) {
        Step(u, dudt, dudt2, t, dt);
    }

    /// Get current step number
    int CurrentStep() const { return step_; }

    /// Check if this is the first step
    bool IsFirstStep() const { return first_step_; }

protected:
    int step_ = 0;
    bool first_step_ = true;
};


// =============================================================================
// NewmarkCentralDifference
// =============================================================================

/**
 * @class NewmarkCentralDifference
 * @brief Newmark-beta method with beta=0, gamma=0.5 (central difference)
 *
 * This is an explicit, conditionally stable integrator.
 * Stability requires: dt <= CFL * h_min / c_max
 *
 * Update equations:
 *   u^{n+1} = u^n + dt * v^n + 0.5 * dt^2 * a^n
 *   v^{n+1/2} = v^n + 0.5 * dt * a^n
 *   a^{n+1} = M^{-1} * F(t^{n+1}, u^{n+1})
 *   v^{n+1} = v^{n+1/2} + 0.5 * dt * a^{n+1}
 */
class NewmarkCentralDifference : public ExplicitTimeIntegrator {
public:
    NewmarkCentralDifference() = default;

    void Init(WaveOperator& op, real_t t0, real_t dt) override;

    void Step(ParGridFunction& u,
              ParGridFunction& dudt,
              ParGridFunction& dudt2,
              real_t& t, real_t dt) override;

    void AdjointStep(ParGridFunction& u,
                     ParGridFunction& dudt,
                     ParGridFunction& dudt2,
                     real_t& t, real_t dt) override;

    /// Shared Newmark update body. @p adjoint selects between forward
    /// ExplicitSolve and AdjointExplicitSolve; everything else is identical.
    ///
    /// Intentionally public (logically private): nvcc with
    /// --expt-extended-lambda forbids defining an extended __host__
    /// __device__ lambda inside a function with private or protected
    /// access. StepImpl uses MFEM_FORALL internally, so it has to live
    /// in the public section. Step / AdjointStep remain the supported
    /// entry points.
    void StepImpl(ParGridFunction& u,
                  ParGridFunction& dudt,
                  ParGridFunction& dudt2,
                  real_t& t, real_t dt, bool adjoint);

private:
    WaveOperator* op_ = nullptr;
    real_t dt_ = 0.0;
    static constexpr real_t beta_ = 0.0_r;
    static constexpr real_t gamma_ = 0.5_r;
    static constexpr real_t half_ = 0.5_r;
};


}  // namespace SEM

#endif  // SEM_TIME_INTEGRATOR_HPP
