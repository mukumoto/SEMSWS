/**
 * @file AcousticOperator2D.cpp
 * @brief Implementation of 2D acoustic wave operator
 */

#include "operator/acoustic/AcousticOperator2D.hpp"
#include "integ/SEMVisco_IsotropicAcousticIntegrator.hpp"
#include "integ/StiffnessIntegratorFactory.hpp"
#include "util/Profiler.hpp"

namespace SEM {

AcousticOperator2D::AcousticOperator2D(
    ParFiniteElementSpace& fes,
    int order,
    const AcousticMaterialBase2D& material)
    : WaveOperatorBase2D(fes, order), material_(material)
{
}

AcousticOperator2D& AcousticOperator2D::SetupMass() {
    int ne = fes_.GetMesh()->GetNE();
    int ngll = order_ + 1;
    int total = ne * ngll * ngll;

    // Copy inverse kappa to mass coefficient
    // Note: For viscoacoustic, material_.Kappa() is already the unrelaxed value
    // (corrected in IsotropicAcousticMaterial::ApplyAttenuationCorrection())
    Vector inv_kappa_coef(total);
    const real_t* kappa_src = material_.Kappa().Data().HostRead();
    real_t* inv_kappa_dst = inv_kappa_coef.HostWrite();

    for (int i = 0; i < total; i++) {
        inv_kappa_dst[i] = 1.0 / kappa_src[i];
    }

    AssembleDiagonalMass(inv_kappa_coef, false);  // is_vector=false for acoustic
    return *this;
}

AcousticOperator2D& AcousticOperator2D::SetupStiffness() {
    auto integ = CreateAcousticStiffnessIntegrator2D(material_, fes_, dt_);
    domain_integs_.Append(integ.release());
    is_setup_complete_ = true;
    return *this;
}

AcousticOperator2D& AcousticOperator2D::SetupDirichletBC(
    const Array<int>& tdof_list)
{
    dirichlet_tdof_ = tdof_list;
    return *this;
}

void AcousticOperator2D::SetDirichletLDofs(const Array<int>& ldof_list)
{
    dirichlet_ldof_ = ldof_list;
}

void AcousticOperator2D::ApplyBoundaryConditions() {
    // No longer needed: EnforceDirichletBC is called before ExplicitSolve
    // (kept as empty override to satisfy base class interface)
}

void AcousticOperator2D::EnforceDirichletBC(
    ParGridFunction& u,
    ParGridFunction& dudt,
    ParGridFunction& dudt2)
{
    if (dirichlet_ldof_.Size() == 0) return;

    // Use local DOF indices (not true DOF) for ParGridFunction
    u.SetSubVector(dirichlet_ldof_, 0.0);
    dudt.SetSubVector(dirichlet_ldof_, 0.0);
    dudt2.SetSubVector(dirichlet_ldof_, 0.0);
}

void AcousticOperator2D::ResetPhysicsState() {
    // Reset viscoacoustic state if enabled
    if (HasViscoelasticity() && domain_integs_.Size() > 0) {
        auto* integ = static_cast<SEMIntegratorBase2D*>(domain_integs_[0]);
        integ->ResetAttenuationState();
    }
}

bool AcousticOperator2D::HasViscoelasticity() const {
    return material_.HasAttenuation();
}

void AcousticOperator2D::PrintInfo(std::ostream& os) const {
    os << "AcousticOperator2D:\n";
    os << "  Local DOFs: " << fes_.GetTrueVSize() << "\n";
    os << "  Local Elements: " << fes_.GetParMesh()->GetNE() << "\n";
    os << "  Domain integrators: " << domain_integs_.Size() << "\n";
    os << "  Cerjan taper: " << (has_cerjan_taper_ ? "enabled" : "disabled") << "\n";
    if (has_cerjan_taper_ && damping_config_.stats.computed) {
        os << "    Thickness: " << damping_config_.abc_lengths[0] << " m\n";
        os << "    Alpha: " << damping_config_.alpha[0] << "\n";
        os << "    Min f: " << damping_config_.stats.min_f << " (boundary)\n";
        os << "    Max f: " << damping_config_.stats.max_f << " (interior)\n";
    }
    os << "  Viscoacoustic: " << (HasViscoelasticity() ? "enabled (Newmark)" : "disabled") << "\n";
    os << "  Dirichlet DOFs: " << dirichlet_tdof_.Size() << "\n";
}

// =============================================================================
// Checkpointing: attenuation state pack/unpack
// =============================================================================

int AcousticOperator2D::AttenuationStateSize() const {
    if (!HasViscoelasticity() || domain_integs_.Size() == 0) return 0;

    auto* visco = static_cast<SEMVisco_IsotropicAcousticIntegrator2D*>(domain_integs_[0]);
    auto* mem = visco->GetAttenuation().GetMemory();
    if (!mem || !mem->IsInitialized()) return 0;

    // Pack: e1 [ngll*ngll*ne*n_units] + dot_e1_old [ngll*ngll*ne]
    return mem->e1.Size() + mem->dot_e1_old.Size();
}

void AcousticOperator2D::GetAttenuationState(Vector& state) const {
    if (!HasViscoelasticity() || domain_integs_.Size() == 0) return;

    auto* visco = static_cast<SEMVisco_IsotropicAcousticIntegrator2D*>(domain_integs_[0]);
    auto* mem = visco->GetAttenuation().GetMemory();
    if (!mem || !mem->IsInitialized()) return;

    int e1_size = mem->e1.Size();
    int dot_size = mem->dot_e1_old.Size();
    state.SetSize(e1_size + dot_size);

    // GPU-native pack: read/write device pointers, copy with MFEM_FORALL
    const real_t* e1_src = mem->e1.Read();
    const real_t* dot_src = mem->dot_e1_old.Read();
    state.UseDevice(true);
    real_t* dst = state.Write();

    MFEM_FORALL(i, e1_size, { dst[i] = e1_src[i]; });
    MFEM_FORALL(i, dot_size, { dst[e1_size + i] = dot_src[i]; });
}

void AcousticOperator2D::SetAttenuationState(const Vector& state) {
    if (!HasViscoelasticity() || domain_integs_.Size() == 0) return;

    auto* visco = static_cast<SEMVisco_IsotropicAcousticIntegrator2D*>(domain_integs_[0]);
    auto* mem = visco->GetAttenuation().GetMemory();
    if (!mem || !mem->IsInitialized()) return;

    int e1_size = mem->e1.Size();
    int dot_size = mem->dot_e1_old.Size();
    MFEM_VERIFY(state.Size() == e1_size + dot_size,
        "Attenuation state size mismatch: got " << state.Size()
        << ", expected " << e1_size + dot_size);

    // GPU-native unpack: all on device, no host transfer
    const real_t* src = state.Read();
    real_t* e1_dst = mem->e1.Write();
    real_t* dot_dst = mem->dot_e1_old.Write();

    MFEM_FORALL(i, e1_size, { e1_dst[i] = src[i]; });
    MFEM_FORALL(i, dot_size, { dot_dst[i] = src[e1_size + i]; });
}

}  // namespace SEM
