/**
 * @file SimulationRunner.cpp
 * @brief Implementation of SimulationRunner template class
 */

#include "simulation/SimulationRunner.hpp"
#include <iostream>

namespace SEM {

// =============================================================================
// Constructor
// =============================================================================

template<int Dim>
SimulationRunner<Dim>::SimulationRunner(MPI_Comm comm)
    : comm_(comm)
{
    MPI_Comm_rank(comm_, &rank_);
}

// =============================================================================
// Dependencies
// =============================================================================

template<int Dim>
void SimulationRunner<Dim>::SetOperator(WaveOperator* op) {
    op_ = op;
}

template<int Dim>
void SimulationRunner<Dim>::SetIntegrator(ExplicitTimeIntegrator* integrator) {
    integrator_ = integrator;
}


// =============================================================================
// Execution
// =============================================================================

template<int Dim>
void SimulationRunner<Dim>::Initialize(ParGridFunction& u, ParGridFunction& v,
                                        ParGridFunction& a) {
    MFEM_VERIFY(op_ != nullptr, "Operator must be set before initialization");
    MFEM_VERIFY(integrator_ != nullptr, "Integrator must be set before initialization");

    // Reset state vectors
    u = 0.0;
    v = 0.0;
    a = 0.0;

    // Initialize time integrator
    state_.time = t0_;
    state_.step = 0;
    state_.dt = dt_;
    state_.total_steps = nt_;
    state_.is_finished = false;
    state_.is_initialized = true;

    integrator_->Init(*op_, t0_, dt_);

    // Reset operator state
    op_->ResetState();
}

template<int Dim>
bool SimulationRunner<Dim>::Step(ParGridFunction& u, ParGridFunction& v,
                                  ParGridFunction& a) {
    if (state_.is_finished) {
        return false;
    }

    // Time step (step index is 0-based: step=0,1,...,nt-1)
    integrator_->Step(u, v, a, state_.time, dt_);

    // Update time (but NOT step yet - callbacks use current step index)
    state_.time += dt_;

    // Call callbacks BEFORE incrementing step
    // This ensures receivers record at current step index (0-based)
    if (step_callback_) {
        step_callback_(state_, u, v, a);
    }

    if (progress_callback_ && progress_interval_ > 0 && state_.step % progress_interval_ == 0) {
        progress_callback_(state_.step, nt_, state_.time);
    }

    // Increment step AFTER callbacks
    state_.step++;

    // Check if finished (we've done nt_ steps: 0, 1, ..., nt_-1)
    if (state_.step >= nt_) {
        state_.is_finished = true;
    }

    return !state_.is_finished;
}


template<int Dim>
void SimulationRunner<Dim>::Reset(ParGridFunction& u, ParGridFunction& v,
                                   ParGridFunction& a, WaveOperator* op) {
    u = 0.0;
    v = 0.0;
    a = 0.0;

    state_.step = 0;
    state_.time = t0_;
    state_.is_finished = false;
    state_.is_initialized = false;

    if (op) {
        op->ResetState();
    }
}

// =============================================================================
// Callbacks
// =============================================================================

template<int Dim>
void SimulationRunner<Dim>::SetStepCallback(StepCallback callback) {
    step_callback_ = std::move(callback);
}

template<int Dim>
void SimulationRunner<Dim>::SetProgressCallback(ProgressCallback callback, int interval) {
    progress_callback_ = std::move(callback);
    progress_interval_ = interval;
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

template class SimulationRunner<2>;
template class SimulationRunner<3>;

}  // namespace SEM
