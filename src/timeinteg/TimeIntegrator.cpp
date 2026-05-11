/**
 * @file TimeIntegrator.cpp
 * @brief Implementation of time integrators
 */

#include "timeinteg/TimeIntegrator.hpp"
#include "general/forall.hpp"
#include "util/Profiler.hpp"

namespace SEM {

// =============================================================================
// NewmarkCentralDifference
// =============================================================================

void NewmarkCentralDifference::Init(WaveOperator& op, real_t t0, real_t dt) {
    op_ = &op;
    dt_ = dt;
    first_step_ = true;
    step_ = 0;
    op_->SetTime(t0);
}

void NewmarkCentralDifference::Step(
    ParGridFunction& u,
    ParGridFunction& dudt,
    ParGridFunction& dudt2,
    real_t& t, real_t dt)
{
    StepImpl(u, dudt, dudt2, t, dt, /*adjoint=*/false);
}

void NewmarkCentralDifference::AdjointStep(
    ParGridFunction& u,
    ParGridFunction& dudt,
    ParGridFunction& dudt2,
    real_t& t, real_t dt)
{
    StepImpl(u, dudt, dudt2, t, dt, /*adjoint=*/true);
}

void NewmarkCentralDifference::StepImpl(
    ParGridFunction& u,
    ParGridFunction& dudt,
    ParGridFunction& dudt2,
    real_t& t, real_t dt, bool adjoint)
{
    if (first_step_) {
        // First step: zero initial conditions on GPU (not host!)
        // Host zeroing invalidates GPU data, causing GPU kernels to read stale zeros
        {
            PROFILE_REGION_GPU("InitializeFields");
            const int N = u.Size();
            auto u_data = u.Write();
            auto v_data = dudt.Write();
            auto a_data = dudt2.Write();
            mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
                u_data[i] = 0.0;
                v_data[i] = 0.0;
                a_data[i] = 0.0;
            });
        }

        first_step_ = false;
    } else {
        // Standard Newmark central difference update
        // Note: SetTime(t+dt) sets the time for the source evaluation
        // The caller should update t AFTER this step returns to match legacy behavior
        op_->SetTime(t + dt);

        // Fused kernel: update displacement and predictor step
        // u += dt * dudt + 0.5 * dt^2 * dudt2
        // dudt += 0.5 * dt * dudt2 (predictor)
        {
            PROFILE_REGION_GPU("UpdateAndPredict");
            const int N = u.Size();
            auto u_data = u.ReadWrite();
            auto v_data = dudt.ReadWrite();
            const auto a_data = dudt2.Read();
            const real_t dt_val = dt;
            const real_t half_dt = half_ * dt;
            const real_t half_dt2 = half_ * dt * dt;
            mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
                u_data[i] += dt_val * v_data[i] + half_dt2 * a_data[i];
                v_data[i] += half_dt * a_data[i];
            });
        }

        // Enforce Dirichlet BC before stiffness assembly
        // Ensures K*u reads zero at Dirichlet DOFs
        op_->EnforceDirichletBC(u, dudt, dudt2);

        // Compute new acceleration (ExplicitSolve has its own profiling regions)
        if (adjoint) {
            op_->AdjointExplicitSolve(u, dudt, dudt2, dt);
        } else {
            op_->ExplicitSolve(u, dudt, dudt2, dt);
        }

        // dudt += 0.5 * dt * dudt2 (corrector step, GPU kernel)
        {
            PROFILE_REGION_GPU("CorrectorStep");
            const int N = dudt.Size();
            auto v_data = dudt.ReadWrite();
            const auto a_data = dudt2.Read();
            const real_t half_dt = half_ * dt;
            mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
                v_data[i] += half_dt * a_data[i];
            });
        }

        // Apply Cerjan taper after complete Newmark step (no-op if not enabled)
        op_->ApplyCerjanTaper(u, dudt, dudt2);

        // Enforce Dirichlet BC on state vectors (no-op if no Dirichlet DOFs)
        op_->EnforceDirichletBC(u, dudt, dudt2);
    }

    // Note: t is NOT incremented here to match legacy Newmark behavior.
    // The caller should update t after each step using: t = step_index * dt
    step_++;
}


}  // namespace SEM
