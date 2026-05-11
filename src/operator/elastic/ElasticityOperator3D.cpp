/**
 * @file ElasticityOperator3D.cpp
 * @brief Implementation of 3D elastic wave operator
 */

#include "operator/elastic/ElasticityOperator3D.hpp"
#include "material/isotropic_elastic/IsotropicElasticMaterial.hpp"
#include "integ/StiffnessIntegratorFactory.hpp"
#include "integ/SEMVisco_IsotropicElasticIntegrator.hpp"
#include "integ/attenuation/AttenuationModel.hpp"
#include "util/Profiler.hpp"

namespace SEM {

ElasticOperator3D::ElasticOperator3D(
    ParFiniteElementSpace& fes,
    int order,
    const ElasticMaterialBase3D& material)
    : WaveOperatorBase3D(fes, order), material_(material)
{
    // Enable GPU memory for stiffness kernel output
    rhs_->UseDevice(true);
}

ElasticOperator3D& ElasticOperator3D::SetupMass() {
    int ne = fes_.GetMesh()->GetNE();
    int ngll = order_ + 1;
    const int64_t total = (int64_t)ne * ngll * ngll * ngll;
    MFEM_VERIFY(total <= INT_MAX,
                "ElasticOperator3D::SetupMass size (" << total << ") exceeds int32 limit. "
                "Reduce elements per GPU (ne=" << ne << ") or increase GPU count.");

    // Copy density to mass coefficient
    // Use HostRead() to ensure host memory is valid (not GetData which may be stale on GPU)
    Vector rho_coef(total);
    const real_t* rho_src = material_.Rho().Data().HostRead();
    real_t* rho_dst = rho_coef.HostWrite();
    for (int i = 0; i < total; i++) {
        rho_dst[i] = rho_src[i];
    }

    AssembleDiagonalMass(rho_coef, true);  // is_vector=true for elastic

    // Enable device memory for GPU kernels
    M_inv_->UseDevice(true);
    return *this;
}

ElasticOperator3D& ElasticOperator3D::SetupStiffness() {
    PROFILE_REGION("Setup:SetupStiffness");
    auto integ = CreateElasticStiffnessIntegrator3D(material_, fes_, dt_);
    domain_integs_.Append(integ.release());
    is_setup_complete_ = true;
    return *this;
}

void ElasticOperator3D::ResetPhysicsState() {
    // Reset viscoelastic state if enabled
    if (HasViscoelasticity() && domain_integs_.Size() > 0) {
        auto* integ = static_cast<SEMIntegratorBase3D*>(domain_integs_[0]);
        integ->ResetAttenuationState();
    }
}

bool ElasticOperator3D::HasViscoelasticity() const {
    return material_.HasAttenuation();
}

void ElasticOperator3D::PrintInfo(std::ostream& os) const {
    os << "ElasticOperator3D:\n";
    os << "  Local DOFs: " << fes_.GetTrueVSize() << "\n";
    os << "  Local Elements: " << fes_.GetParMesh()->GetNE() << "\n";
    os << "  Material type: ";
    switch (material_.GetType()) {
        case MaterialType::IsotropicElastic: os << "IsotropicElastic"; break;
        case MaterialType::AnisotropicElastic: os << "AnisotropicElastic"; break;
        // Future: VTI case will be added when MaterialType::VTI is defined
        default: os << "Unknown"; break;
    }
    os << "\n";
    os << "  Domain integrators: " << domain_integs_.Size() << "\n";
    os << "  Cerjan taper: " << (has_cerjan_taper_ ? "enabled" : "disabled") << "\n";
    if (has_cerjan_taper_ && damping_config_.stats.computed) {
        os << "    Thickness: " << damping_config_.abc_lengths[0] << " m\n";
        os << "    Alpha: " << damping_config_.alpha[0] << "\n";
        os << "    Min f: " << damping_config_.stats.min_f << " (boundary)\n";
        os << "    Max f: " << damping_config_.stats.max_f << " (interior)\n";
    }
    os << "  Viscoelasticity: " << (HasViscoelasticity() ? "enabled" : "disabled") << "\n";
}

// =============================================================================
// Checkpointing: viscoelastic memory state pack/unpack (3D, 6 components)
// =============================================================================
//
// State layout matches ViscoelasticMemory3D:
//   M_packed          [6 · ngllx · nglly · ngllz · ne · n_units]
//   strain_old_packed [6 · ngllx · nglly · ngllz · ne]
// Same serialization contract as ElasticOperator2D; only the per-element
// component count differs (3 → 6).

int ElasticOperator3D::AttenuationStateSize() const {
    if (!HasViscoelasticity() || domain_integs_.Size() == 0) return 0;

    auto* visco = static_cast<SEMVisco_IsotropicElasticIntegrator3D*>(domain_integs_[0]);
    const auto* mem = visco->GetAttenuation().GetMemory();
    if (!mem || !mem->IsInitialized()) return 0;

    return mem->M_packed.Size() + mem->strain_old_packed.Size();
}

void ElasticOperator3D::GetAttenuationState(Vector& state) const {
    if (!HasViscoelasticity() || domain_integs_.Size() == 0) return;

    auto* visco = static_cast<SEMVisco_IsotropicElasticIntegrator3D*>(domain_integs_[0]);
    const auto* mem = visco->GetAttenuation().GetMemory();
    if (!mem || !mem->IsInitialized()) return;

    const int m_size = mem->M_packed.Size();
    const int s_size = mem->strain_old_packed.Size();
    state.SetSize(m_size + s_size);

    const real_t* m_src = mem->M_packed.Read();
    const real_t* s_src = mem->strain_old_packed.Read();
    state.UseDevice(true);
    real_t* dst = state.Write();

    MFEM_FORALL(i, m_size, { dst[i] = m_src[i]; });
    MFEM_FORALL(i, s_size, { dst[m_size + i] = s_src[i]; });
}

void ElasticOperator3D::SetAttenuationState(const Vector& state) {
    if (!HasViscoelasticity() || domain_integs_.Size() == 0) return;

    auto* visco = static_cast<SEMVisco_IsotropicElasticIntegrator3D*>(domain_integs_[0]);
    auto* mem = visco->GetAttenuation().GetMemory();
    if (!mem || !mem->IsInitialized()) return;

    const int m_size = mem->M_packed.Size();
    const int s_size = mem->strain_old_packed.Size();
    MFEM_VERIFY(state.Size() == m_size + s_size,
        "Attenuation state size mismatch: got " << state.Size()
        << ", expected " << m_size + s_size);

    const real_t* src = state.Read();
    real_t* m_dst = mem->M_packed.Write();
    real_t* s_dst = mem->strain_old_packed.Write();

    MFEM_FORALL(i, m_size, { m_dst[i] = src[i]; });
    MFEM_FORALL(i, s_size, { s_dst[i] = src[m_size + i]; });
}

}  // namespace SEM
