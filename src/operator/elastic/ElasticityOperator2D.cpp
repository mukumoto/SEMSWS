/**
 * @file ElasticityOperator2D.cpp
 * @brief Implementation of 2D elastic wave operator
 */

#include "operator/elastic/ElasticityOperator2D.hpp"
#include "material/isotropic_elastic/IsotropicElasticMaterial.hpp"
#include "integ/StiffnessIntegratorFactory.hpp"
#include "integ/SEMVisco_IsotropicElasticIntegrator.hpp"
#include "integ/attenuation/AttenuationModel.hpp"
#include "util/Profiler.hpp"

namespace SEM {

ElasticOperator2D::ElasticOperator2D(
    ParFiniteElementSpace& fes,
    int order,
    const ElasticMaterialBase2D& material)
    : WaveOperatorBase2D(fes, order), material_(material)
{
}

ElasticOperator2D& ElasticOperator2D::SetupMass() {
    int ne = fes_.GetMesh()->GetNE();
    int ngll = order_ + 1;
    int total = ne * ngll * ngll;

    // Copy density to mass coefficient
    // Use HostRead() to ensure host memory is valid (not GetData which may be stale on GPU)
    Vector rho_coef(total);
    const real_t* rho_src = material_.Rho().Data().HostRead();
    real_t* rho_dst = rho_coef.HostWrite();
    for (int i = 0; i < total; i++) {
        rho_dst[i] = rho_src[i];
    }

    AssembleDiagonalMass(rho_coef, true);  // is_vector=true for elastic
    return *this;
}

ElasticOperator2D& ElasticOperator2D::SetupStiffness() {
    auto integ = CreateElasticStiffnessIntegrator2D(material_, fes_, dt_);
    domain_integs_.Append(integ.release());
    is_setup_complete_ = true;
    return *this;
}

void ElasticOperator2D::ResetPhysicsState() {
    // Reset viscoelastic state if enabled
    if (HasViscoelasticity() && domain_integs_.Size() > 0) {
        auto* integ = static_cast<SEMIntegratorBase2D*>(domain_integs_[0]);
        integ->ResetAttenuationState();
    }
}

bool ElasticOperator2D::HasViscoelasticity() const {
    return material_.HasAttenuation();
}

void ElasticOperator2D::PrintInfo(std::ostream& os) const {
    os << "ElasticOperator2D:\n";
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
// Checkpointing: viscoelastic memory state pack/unpack
// =============================================================================
//
// State layout matches ViscoelasticMemory2D:
//   M_packed          [3 · ngllx · nglly · ne · n_units]
//   strain_old_packed [3 · ngllx · nglly · ne]
// Concatenated into a single flat Vector so Revolve can treat attenuation
// as an opaque blob. Mirrors AcousticOperator2D::{Get,Set}AttenuationState.

int ElasticOperator2D::AttenuationStateSize() const {
    if (!HasViscoelasticity() || domain_integs_.Size() == 0) return 0;

    auto* visco = static_cast<SEMVisco_IsotropicElasticIntegrator2D*>(domain_integs_[0]);
    const auto* mem = visco->GetAttenuation().GetMemory();
    if (!mem || !mem->IsInitialized()) return 0;

    return mem->M_packed.Size() + mem->strain_old_packed.Size();
}

void ElasticOperator2D::GetAttenuationState(Vector& state) const {
    if (!HasViscoelasticity() || domain_integs_.Size() == 0) return;

    auto* visco = static_cast<SEMVisco_IsotropicElasticIntegrator2D*>(domain_integs_[0]);
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

void ElasticOperator2D::SetAttenuationState(const Vector& state) {
    if (!HasViscoelasticity() || domain_integs_.Size() == 0) return;

    auto* visco = static_cast<SEMVisco_IsotropicElasticIntegrator2D*>(domain_integs_[0]);
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
