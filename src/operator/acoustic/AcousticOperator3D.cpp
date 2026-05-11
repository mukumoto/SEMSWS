/**
 * @file AcousticOperator3D.cpp
 * @brief Implementation of 3D acoustic wave operator
 */

#include "operator/acoustic/AcousticOperator3D.hpp"
#include "integ/StiffnessIntegratorFactory.hpp"
#include "integ/SEMVisco_IsotropicAcousticIntegrator.hpp"
#include "util/Profiler.hpp"

namespace SEM {

AcousticOperator3D::AcousticOperator3D(
    ParFiniteElementSpace& fes,
    int order,
    const AcousticMaterialBase3D& material)
    : WaveOperatorBase3D(fes, order), material_(material)
{
}

AcousticOperator3D& AcousticOperator3D::SetupMass() {
    int ne = fes_.GetMesh()->GetNE();
    int ngll = order_ + 1;
    int total = ne * ngll * ngll * ngll;

    // Copy inverse kappa to mass coefficient
    // Use HostRead() to ensure host memory is valid (not GetData which may be stale on GPU)
    Vector inv_kappa_coef(total);
    const real_t* kappa_src = material_.Kappa().Data().HostRead();
    real_t* inv_kappa_dst = inv_kappa_coef.HostWrite();
    for (int i = 0; i < total; i++) {
        inv_kappa_dst[i] = 1.0 / kappa_src[i];
    }

    AssembleDiagonalMass(inv_kappa_coef, false);  // is_vector=false for acoustic
    return *this;
}

AcousticOperator3D& AcousticOperator3D::SetupStiffness() {
    auto integ = CreateAcousticStiffnessIntegrator3D(material_, fes_, dt_);
    domain_integs_.Append(integ.release());
    is_setup_complete_ = true;
    return *this;
}

AcousticOperator3D& AcousticOperator3D::SetupDirichletBC(
    const Array<int>& tdof_list)
{
    dirichlet_tdof_ = tdof_list;
    return *this;
}

void AcousticOperator3D::SetDirichletLDofs(const Array<int>& ldof_list)
{
    dirichlet_ldof_ = ldof_list;
}

void AcousticOperator3D::ApplyBoundaryConditions() {
    // No longer needed: EnforceDirichletBC is called before ExplicitSolve
}

void AcousticOperator3D::EnforceDirichletBC(
    ParGridFunction& u,
    ParGridFunction& dudt,
    ParGridFunction& dudt2)
{
    if (dirichlet_ldof_.Size() == 0) return;

    u.SetSubVector(dirichlet_ldof_, 0.0);
    dudt.SetSubVector(dirichlet_ldof_, 0.0);
    dudt2.SetSubVector(dirichlet_ldof_, 0.0);
}

// =============================================================================
// Viscoelasticity hooks
// =============================================================================

bool AcousticOperator3D::HasViscoelasticity() const {
    return material_.HasAttenuation();
}

void AcousticOperator3D::ResetPhysicsState() {
    // Reset viscoacoustic memory variables (e1, dot_e1_old) between sources.
    if (HasViscoelasticity() && domain_integs_.Size() > 0) {
        auto* integ = static_cast<SEMIntegratorBase3D*>(domain_integs_[0]);
        integ->ResetAttenuationState();
    }
}

// =============================================================================
// Checkpointing: attenuation state pack/unpack
// Mirrors AcousticOperator2D; ViscoacousticMemory3D uses the same e1 /
// dot_e1_old layout so the packing logic is identical up to the 3D view.
// =============================================================================

int AcousticOperator3D::AttenuationStateSize() const {
    if (!HasViscoelasticity() || domain_integs_.Size() == 0) return 0;

    auto* visco = static_cast<SEMVisco_IsotropicAcousticIntegrator3D*>(domain_integs_[0]);
    auto* mem = visco->GetAttenuation().GetMemory();
    if (!mem || !mem->IsInitialized()) return 0;

    return mem->e1.Size() + mem->dot_e1_old.Size();
}

void AcousticOperator3D::GetAttenuationState(Vector& state) const {
    if (!HasViscoelasticity() || domain_integs_.Size() == 0) return;

    auto* visco = static_cast<SEMVisco_IsotropicAcousticIntegrator3D*>(domain_integs_[0]);
    auto* mem = visco->GetAttenuation().GetMemory();
    if (!mem || !mem->IsInitialized()) return;

    const int e1_size = mem->e1.Size();
    const int dot_size = mem->dot_e1_old.Size();
    state.SetSize(e1_size + dot_size);

    const real_t* e1_src = mem->e1.Read();
    const real_t* dot_src = mem->dot_e1_old.Read();
    state.UseDevice(true);
    real_t* dst = state.Write();

    MFEM_FORALL(i, e1_size, { dst[i] = e1_src[i]; });
    MFEM_FORALL(i, dot_size, { dst[e1_size + i] = dot_src[i]; });
}

void AcousticOperator3D::SetAttenuationState(const Vector& state) {
    if (!HasViscoelasticity() || domain_integs_.Size() == 0) return;

    auto* visco = static_cast<SEMVisco_IsotropicAcousticIntegrator3D*>(domain_integs_[0]);
    auto* mem = visco->GetAttenuation().GetMemory();
    if (!mem || !mem->IsInitialized()) return;

    const int e1_size = mem->e1.Size();
    const int dot_size = mem->dot_e1_old.Size();
    MFEM_VERIFY(state.Size() == e1_size + dot_size,
        "Attenuation state size mismatch: got " << state.Size()
        << ", expected " << e1_size + dot_size);

    const real_t* src = state.Read();
    real_t* e1_dst = mem->e1.Write();
    real_t* dot_dst = mem->dot_e1_old.Write();

    MFEM_FORALL(i, e1_size, { e1_dst[i] = src[i]; });
    MFEM_FORALL(i, dot_size, { dot_dst[i] = src[e1_size + i]; });
}

void AcousticOperator3D::PrintInfo(std::ostream& os) const {
    os << "AcousticOperator3D:\n";
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
    os << "  Dirichlet DOFs: " << dirichlet_tdof_.Size() << "\n";
}

}  // namespace SEM
