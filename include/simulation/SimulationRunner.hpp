/**
 * @file SimulationRunner.hpp
 * @brief Time integration loop management
 *
 * SimulationRunner handles:
 * - Time stepping parameters (dt, nt, t0)
 * - Integration loop (Initialize, Step, Run, Reset)
 * - Callbacks (step, progress)
 * - SimulationState tracking
 *
 * This class operates on state vectors passed by reference (does not own them).
 */

#ifndef SEM_SIMULATION_RUNNER_HPP
#define SEM_SIMULATION_RUNNER_HPP

#include <mfem.hpp>
#include <functional>

#include "operator/Operator.hpp"
#include "timeinteg/TimeIntegrator.hpp"

namespace SEM {


using namespace mfem;

// =============================================================================
// SimulationState
// =============================================================================

/**
 * @struct SimulationState
 * @brief Current state of simulation
 */
struct SimulationState {
    int step = 0;
    real_t time = 0.0;
    real_t dt = 0.0;
    int total_steps = 0;
    bool is_finished = false;
    bool is_initialized = false;
};

// =============================================================================
// Callback Types
// =============================================================================

/// Callback for per-step processing
using StepCallback = std::function<void(const SimulationState& state,
                                        const ParGridFunction& u,
                                        const ParGridFunction& v,
                                        const ParGridFunction& a)>;

/// Callback for progress reporting
using ProgressCallback = std::function<void(int step, int total, real_t time)>;

// =============================================================================
// SimulationRunner
// =============================================================================

/**
 * @class SimulationRunner
 * @brief Manages time integration loop for simulation
 * @tparam Dim Spatial dimension (2 or 3) - used for logging only
 *
 * Responsibilities:
 * - Time parameter management (dt, nt, t0)
 * - Time stepping loop (Initialize, Step, Run)
 * - Callback invocation
 * - State tracking
 *
 * This class does NOT own:
 * - WaveOperator (receives raw pointer)
 * - ExplicitTimeIntegrator (receives raw pointer)
 * - State vectors (receives references)
 */
template<int Dim>
class SimulationRunner {
public:
    /**
     * @brief Construct runner with MPI communicator
     * @param comm MPI communicator
     */
    explicit SimulationRunner(MPI_Comm comm);

    /// Destructor
    ~SimulationRunner() = default;

    // -------------------------------------------------------------------------
    // Dependencies (non-owning)
    // -------------------------------------------------------------------------

    /**
     * @brief Set the wave operator (non-owning)
     * @param op Pointer to WaveOperator
     */
    void SetOperator(WaveOperator* op);

    /**
     * @brief Set the time integrator (non-owning)
     * @param integrator Pointer to ExplicitTimeIntegrator
     */
    void SetIntegrator(ExplicitTimeIntegrator* integrator);

    // -------------------------------------------------------------------------
    // Time Parameters
    // -------------------------------------------------------------------------

    /// Set time step
    void SetDt(real_t dt) { dt_ = dt; }

    /// Set number of steps
    void SetNumSteps(int nt) { nt_ = nt; }

    /// Set initial time
    void SetT0(real_t t0) { t0_ = t0; }

    /// Get time step
    real_t Dt() const { return dt_; }

    /// Get number of steps
    int NumSteps() const { return nt_; }

    /// Get initial time
    real_t T0() const { return t0_; }

    // -------------------------------------------------------------------------
    // Execution
    // -------------------------------------------------------------------------

    /**
     * @brief Initialize simulation for time stepping
     * @param u Displacement field
     * @param v Velocity field
     * @param a Acceleration field
     *
     * Resets state vectors to zero and initializes integrator.
     */
    void Initialize(ParGridFunction& u, ParGridFunction& v, ParGridFunction& a);

    /**
     * @brief Execute single time step
     * @param u Displacement field
     * @param v Velocity field
     * @param a Acceleration field
     * @return true if more steps remain, false if finished
     */
    bool Step(ParGridFunction& u, ParGridFunction& v, ParGridFunction& a);


    /**
     * @brief Reset simulation state
     * @param u Displacement field
     * @param v Velocity field
     * @param a Acceleration field
     * @param op Wave operator (optional, for ResetState call)
     */
    void Reset(ParGridFunction& u, ParGridFunction& v, ParGridFunction& a,
               WaveOperator* op = nullptr);

    // -------------------------------------------------------------------------
    // Callbacks
    // -------------------------------------------------------------------------

    /**
     * @brief Set per-step callback
     * @param callback Function called each step
     */
    void SetStepCallback(StepCallback callback);

    /**
     * @brief Set progress callback
     * @param callback Function called for progress reporting
     * @param interval Steps between calls
     */
    void SetProgressCallback(ProgressCallback callback, int interval = 100);

    // -------------------------------------------------------------------------
    // State Access
    // -------------------------------------------------------------------------

    /// Get current simulation state
    const SimulationState& State() const { return state_; }

    /// Check if initialized
    bool IsInitialized() const { return state_.is_initialized; }

    /// Check if finished
    bool IsFinished() const { return state_.is_finished; }

    /// Get current step
    int CurrentStep() const { return state_.step; }

    /// Get current time
    real_t CurrentTime() const { return state_.time; }

    /**
     * @brief Set runner state directly (for checkpoint restore)
     * @param step Step number
     * @param time Current time
     *
     * Used by AdjointSimulation to restore state after checkpoint load.
     */
    void SetState(int step, real_t time) {
        state_.step = step;
        state_.time = time;
        state_.is_finished = (step >= nt_);
    }

    // -------------------------------------------------------------------------
    // Query
    // -------------------------------------------------------------------------

    /// Get MPI communicator
    MPI_Comm Comm() const { return comm_; }

    /// Check if root process
    bool IsRoot() const { return rank_ == 0; }

private:
    MPI_Comm comm_;
    int rank_ = 0;

    // Non-owning pointers to external objects
    WaveOperator* op_ = nullptr;
    ExplicitTimeIntegrator* integrator_ = nullptr;

    // Time parameters
    real_t dt_ = 0.001;
    real_t t0_ = 0.0;
    int nt_ = 1000;

    // State
    SimulationState state_;

    // Callbacks
    StepCallback step_callback_;
    ProgressCallback progress_callback_;
    int progress_interval_ = 100;
};

// Type aliases
using SimulationRunner2D = SimulationRunner<2>;
using SimulationRunner3D = SimulationRunner<3>;

}  // namespace SEM

#endif  // SEM_SIMULATION_RUNNER_HPP
