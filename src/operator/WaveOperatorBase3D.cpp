/**
 * @file WaveOperatorBase3D.cpp
 * @brief Implementation of 3D wave operator base class
 */

#include "operator/WaveOperatorBase3D.hpp"
#include "operator/damping/CerjanABC.hpp"
#include "srcrecv/Source.hpp"
#include "general/forall.hpp"
#include "util/Profiler.hpp"

namespace SEM {

WaveOperatorBase3D::WaveOperatorBase3D(ParFiniteElementSpace& fes, int order)
    : WaveOperator(fes), order_(order)
{
    rhs_vec_.SetSize(fes_.GetTrueVSize());
    rhs_vec_.UseDevice(true);
    rhs_ = std::make_unique<ParGridFunction>(&fes_);
    rhs_->UseDevice(true);
}

WaveOperatorBase3D::~WaveOperatorBase3D() {
    for (int i = 0; i < domain_integs_.Size(); i++) {
        delete domain_integs_[i];
    }
    domain_integs_.DeleteAll();
}

WaveOperator& WaveOperatorBase3D::SetupSource(PointSourceCollection& source) {
    source_ = &source;
    return *this;
}

void WaveOperatorBase3D::ResetState() {
    current_step_ = 0;
    ResetPhysicsState();
}

void WaveOperatorBase3D::AssemblePreCouplingRHS(
    const ParGridFunction& u, real_t dt)
{
    PROFILE_REGION("AssemblePreCouplingRHS");

    current_step_ = static_cast<int>(std::round(this->GetTime() / dt));

    // Step 1: Assemble domain RHS (currently stiffness only): rhs_ = -K*u
    {
        PROFILE_REGION_GPU("AssembleDomainRHS");
        AssembleDomainRHS(u);
    }

    // Step 2: Add external source
    if (source_) {
        PROFILE_REGION("SourceAssemble");
        source_->Assemble(current_step_, *rhs_);
    }
    // rhs_ now holds LDOF-level raw RHS. Callers may add surface-integral
    // contributions (coupling, traction BCs, …) before FinalizeAndApplyMass.
}

void WaveOperatorBase3D::FinalizeAndApplyMass(
    const ParGridFunction& dudt, ParGridFunction& dudt2)
{
    PROFILE_REGION("FinalizeAndApplyMass");

    // Step 3: Parallel assembly
    {
        PROFILE_REGION("ParallelAssemble");
        rhs_->ParallelAssemble(rhs_vec_);
    }

    // Step 4: Apply boundary conditions (hook for Acoustic Dirichlet)
    ApplyBoundaryConditions();

    rhs_->SetFromTrueDofs(rhs_vec_);

    // Step 5: Apply damping and inverse mass
    {
        PROFILE_REGION_GPU("ApplyInverseMass");
        ApplyInverseMass(dudt, dudt2);
    }
}

void WaveOperatorBase3D::ExplicitSolve(
    const ParGridFunction& u,
    const ParGridFunction& dudt,
    ParGridFunction& dudt2,
    real_t dt)
{
    // Thin wrapper preserving the pre-refactor monolithic signature. Non-
    // coupled callers (ForwardSimulation, FWI, pure elastic/acoustic tests)
    // hit this path and see exactly the old behaviour; only the coupled
    // facade bypasses this wrapper to insert iface_.Apply*RHS between the
    // two stages (see CoupledSimulationFacade::Step).
    PROFILE_REGION("ExplicitSolve");
    AssemblePreCouplingRHS(u, dt);
    FinalizeAndApplyMass(dudt, dudt2);
}

// =============================================================================
// Protected helper methods
// =============================================================================

void WaveOperatorBase3D::AssembleDiagonalMass(const Vector& coef, bool is_vector) {
    PROFILE_REGION("Setup:AssembleDiagonalMass");

    // Store coefficient for potential damping use
    mass_coef_ = coef;
    // Sync coefficient to GPU
    mass_coef_.UseDevice(true);
    mass_coef_.Read();

    // Create mass integrator and assemble geometry
    mass_integ_ = std::make_unique<SEMMassIntegrator3D>();
    {
        PROFILE_REGION_GPU("Setup:MassAssemblePA");
        mass_integ_->AssemblePA(fes_);
    }
    mass_integ_->SetVectorField(is_vector);

    // Allocate and assemble diagonal mass
    M_inv_ = std::make_unique<ParGridFunction>(&fes_);
    // Zero on GPU (not host!) to maintain GPU-resident data
    {
        const int N = M_inv_->Size();
        M_inv_->UseDevice(true);
        auto M_data = M_inv_->Write();
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
            M_data[i] = 0.0;
        });
    }

    {
        PROFILE_REGION_GPU("Setup:MassAssembleDiagonal");
        mass_integ_->AssembleDiagonalPA(*M_inv_, mass_coef_);
    }

    // Parallel assembly
    Vector M_vec(fes_.GetTrueVSize());
    {
        PROFILE_REGION("Setup:MassParallelAssemble");
        M_inv_->ParallelAssemble(M_vec);
    }
    M_inv_->SetFromTrueDofs(M_vec);

    // Invert mass matrix (diagonal)
    // Use HostReadWrite() to mark host memory as valid for GPU sync
    const int size = M_inv_->Size();
    real_t* M_data = M_inv_->HostReadWrite();

    for (int i = 0; i < size; i++) {
        if (M_data[i] != 0.0) {
            M_data[i] = 1.0 / M_data[i];
        }
    }

    // Enable device memory for GPU kernels (matching SEMSWS pattern)
    // UseDevice(true) after HostReadWrite() ensures proper host→device sync
    M_inv_->UseDevice(true);

    // Force host→device sync for GPU builds
    M_inv_->Read();
}

void WaveOperatorBase3D::AssembleDomainRHS(const ParGridFunction& u) {
    // Assemble domain RHS by summing all registered domain integrators.
    // Currently only the stiffness integrator is registered, and it outputs
    // -K*u directly (no separate negate step needed).
    // Zero on GPU (not host!) to maintain GPU-resident data
    {
        const int N = rhs_->Size();
        auto rhs_data = rhs_->Write();
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
            rhs_data[i] = 0.0;
        });
    }
    for (int i = 0; i < domain_integs_.Size(); i++) {
        domain_integs_[i]->AddMultPA(u, *rhs_);
    }
}

void WaveOperatorBase3D::ApplyInverseMass(
    const ParGridFunction& dudt,
    ParGridFunction& dudt2)
{
    // ü = M⁻¹·(F - K·u)
    const int N = dudt2.Size();
    auto dudt2_data = dudt2.Write();
    const auto rhs_data = rhs_->Read();
    const auto M_data = M_inv_->Read();

    mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
        dudt2_data[i] = rhs_data[i] * M_data[i];
    });
}

void WaveOperatorBase3D::SetupCerjanTaper(const DampingConfig& config) {
    PROFILE_REGION("Setup:CerjanTaper");

    int ngll = order_ + 1;

    // Store damping config for PrintInfo/summary
    damping_config_ = config;

    // Compute Cerjan taper at GLL-point level
    Vector gll_taper;
    damping::ComputeCerjanTaper3D(fes_, config, ngll, gll_taper);

    // Print statistics
    damping::PrintCerjanStats(gll_taper, fes_.GetComm());

    // Project taper into DOF space
    cerjan_taper_ = std::make_unique<ParGridFunction>(&fes_);
    *cerjan_taper_ = 1.0;

    const int vdim = fes_.GetVDim();
    const int ne = fes_.GetMesh()->GetNE();
    const int ndof_per_comp = ngll * ngll * ngll;

    for (int ie = 0; ie < ne; ie++) {
        Array<int> dofs;
        fes_.GetElementVDofs(ie, dofs);

        for (int j = 0; j < ndof_per_comp; j++) {
            real_t tval = gll_taper(ie * ndof_per_comp + j);

            for (int c = 0; c < vdim; c++) {
                int dof_idx = dofs[c * ndof_per_comp + j];
                int abs_dof = (dof_idx >= 0) ? dof_idx : -1 - dof_idx;
                (*cerjan_taper_)(abs_dof) = std::min((*cerjan_taper_)(abs_dof), tval);
            }
        }
    }

    // Sync to device for GPU kernels
    cerjan_taper_->UseDevice(true);
    cerjan_taper_->Read();

    has_cerjan_taper_ = true;
}

void WaveOperatorBase3D::ApplyCerjanTaper(
    ParGridFunction& u, ParGridFunction& dudt, ParGridFunction& dudt2)
{
    if (!has_cerjan_taper_) return;

    PROFILE_REGION_GPU("ApplyCerjanTaper");

    const int N = u.Size();
    auto u_d = u.ReadWrite();
    auto v_d = dudt.ReadWrite();
    auto a_d = dudt2.ReadWrite();
    const auto t_d = cerjan_taper_->Read();

    mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
        u_d[i] *= t_d[i];
        v_d[i] *= t_d[i];
        a_d[i] *= t_d[i];
    });
}

void WaveOperatorBase3D::FreeSetupResources() {
    // Calculate local memory to be freed
    size_t freed_bytes_local = 0;

    if (mass_integ_) {
        freed_bytes_local += mass_integ_->MemoryUsage();
    }
    freed_bytes_local += mass_coef_.Size() * sizeof(real_t);

    // Free resources
    mass_integ_.reset();
    mass_coef_.Destroy();

    // Sum across all processes
    size_t freed_bytes_global = 0;
    MPI_Allreduce(&freed_bytes_local, &freed_bytes_global, 1, MPI_UNSIGNED_LONG,
                  MPI_SUM, fes_.GetComm());

    // Memory freed silently - info available via MemoryTracker if needed
    (void)freed_bytes_global;
}

void WaveOperatorBase3D::DeviceInit() {
    // Initialize device memory for all domain integrators
    for (int i = 0; i < domain_integs_.Size(); i++) {
        auto* sem_integ = static_cast<SEMIntegratorBase3D*>(domain_integs_[i]);
        sem_integ->DeviceInit();
    }

    // Sync mass matrix to device
    if (M_inv_) {
        M_inv_->UseDevice(true);
        M_inv_->Read();
    }

    // Sync Cerjan taper to device
    if (cerjan_taper_) {
        cerjan_taper_->UseDevice(true);
        cerjan_taper_->Read();
    }

    // Sync RHS workspace
    if (rhs_) {
        rhs_->UseDevice(true);
    }
    rhs_vec_.UseDevice(true);
}

}  // namespace SEM
