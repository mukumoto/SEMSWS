/**
 * @file IsotropicElasticSensitivityAD2D.cpp
 * @brief AD sensitivity kernel implementation for 2D isotropic elastic.
 *
 * Seeds λ and μ separately as `mfem::future::dual<real_t, real_t>` and
 * invokes the shared `ElasticStressPhysical2D<T>` template; the material
 * tangent then appears in the `.gradient` component of the returned
 * stress. Contracting that gradient stress tensor with the adjoint
 * strain yields the per-time-step kernel contribution. The ρ path
 * (ü·λ^*) is pointwise and does not use dual.
 *
 * Save() applies the chain rule (λ, μ, ρ) → (Vp, Vs, ρ).
 */

#include "fwi/IsotropicElasticSensitivityAD2D.hpp"
#include "integ/kernels/ElasticFluxKernel2D.hpp"
#include "util/FESOrder.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "common/GpuMacros.hpp"
#include <mfem.hpp>
#include "io/ADIOS2IO.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace SEM {

using Dual = mfem::future::dual<real_t, real_t>;

// =============================================================================
// Constructor / InitCoefficients
// =============================================================================

IsotropicElasticSensitivityAD2D::IsotropicElasticSensitivityAD2D(
    ParFiniteElementSpace& fes,
    const MaterialField& lambda,
    const MaterialField& mu,
    const MaterialField& rho,
    const MaterialField* c_kappa,
    const MaterialField* c_mu)
    : fes_(&fes)
{
    const int order = SafeFESOrder(fes);
    ngll_ = order + 1;
    ne_   = fes.GetParMesh()->GetNE();

    const int total_gll = ngll_ * ngll_ * ne_;

    geom_.Compute(fes);
    dofs_.ComputeVector(fes, order);
    geom_.EnableDevice();
    geom_.SyncToDevice();
    dofs_.EnableDevice();
    dofs_.SyncToDevice();

    auto alloc_dev = [&](Vector& v) {
        v.SetSize(total_gll);
        v.UseDevice(true);
    };
    alloc_dev(lambda_);
    alloc_dev(mu_);
    alloc_dev(coeff_vp_);
    alloc_dev(coeff_vs_mu_);
    alloc_dev(coeff_vs_lam_);
    InitCoefficients(lambda, mu, rho, c_kappa, c_mu);

    alloc_dev(k_lambda_); k_lambda_ = 0.0;
    alloc_dev(k_mu_);     k_mu_     = 0.0;
    alloc_dev(k_rho_);    k_rho_    = 0.0;

    alloc_dev(h_lambda_); h_lambda_ = 0.0;
    alloc_dev(h_mu_);     h_mu_     = 0.0;
    alloc_dev(h_rho_);    h_rho_    = 0.0;
}

void IsotropicElasticSensitivityAD2D::InitCoefficients(
    const MaterialField& lambda,
    const MaterialField& mu,
    const MaterialField& rho,
    const MaterialField* c_kappa,
    const MaterialField* c_mu)
{
    const int total_gll = ngll_ * ngll_ * ne_;

    const real_t* d_lam = lambda.Read();
    const real_t* d_mu  = mu.Read();
    const real_t* d_rho = rho.Read();
    const real_t* d_ck  = c_kappa ? c_kappa->Read() : nullptr;
    const real_t* d_cm  = c_mu    ? c_mu->Read()    : nullptr;

    real_t* d_lambda_out    = lambda_.Write();
    real_t* d_mu_out        = mu_.Write();
    real_t* d_coeff_vp      = coeff_vp_.Write();
    real_t* d_coeff_vs_mu   = coeff_vs_mu_.Write();
    real_t* d_coeff_vs_lam  = coeff_vs_lam_.Write();

    mfem::forall(total_gll, [=] MFEM_HOST_DEVICE (int i) {
        const real_t lam_u = d_lam[i];          // unrelaxed if viscoelastic
        const real_t mu_u  = d_mu[i];
        const real_t r     = d_rho[i];
        const real_t ck    = d_ck ? d_ck[i] : 1.0;
        const real_t cm    = d_cm ? d_cm[i] : 1.0;
        const real_t mu_user  = mu_u / cm;
        // 2D plane strain: κ = λ + μ.
        const real_t kappa_u  = lam_u + mu_u;
        const real_t lam_user = kappa_u / ck - mu_user;
        const real_t vp = sqrt((lam_user + 2.0 * mu_user) / r);
        const real_t vs = sqrt(mu_user / r);
        d_lambda_out[i]    = lam_u;
        d_mu_out[i]        = mu_u;
        // Pure elastic form of chain rule — see comment in
        // IsotropicElasticSensitivity2D.cpp for why c_κ / c_μ must NOT
        // multiply the Save-time coefficients for SLS visco-elastic.
        (void)ck; (void)cm;
        d_coeff_vp[i]      = 2.0 * r * vp;
        d_coeff_vs_mu[i]   = 2.0 * r * vs;
        d_coeff_vs_lam[i]  = -4.0 * r * vs;
    });
}

// =============================================================================
// Reset / ResetHessian
// =============================================================================

void IsotropicElasticSensitivityAD2D::Reset() {
    k_lambda_ = 0.0;
    k_mu_     = 0.0;
    k_rho_    = 0.0;
}

void IsotropicElasticSensitivityAD2D::ResetHessian() {
    h_lambda_ = 0.0;
    h_mu_     = 0.0;
    h_rho_    = 0.0;
}

// =============================================================================
// Accumulate (NGLL dispatch)
// =============================================================================

void IsotropicElasticSensitivityAD2D::Accumulate(
    const Vector& fwd_u, const Vector& fwd_a,
    const Vector& adj_u, real_t dt)
{
    SEM_DISPATCH_NGLL(ngll_, AccumulateKernel, fwd_u, fwd_a, adj_u, dt);
}

// =============================================================================
// AD kernel: two single-seed AD passes (λ, μ) + direct ρ path, fused.
// =============================================================================
//
// Per quadrature point:
//   1. Gather u, λ^*; compute physical gradients.
//   2. Build strain-like quantities (εxx, εxy, εyy) for the adjoint, and
//      the raw physical gradient entries for the forward (ElasticStress
//      uses the full gradient tensor, not the symmetric part).
//   3. λ-seed AD: Dual λ={λ,1}, μ={μ,0}. Call ElasticStressPhysical2D
//      with dual types; extract .gradient of each stress component and
//      contract with ε(λ^*) using the symmetric σ:ε formula
//      (σxx·εxx + σyy·εyy + 2 σxy·εxy).
//   4. μ-seed AD: Dual λ={λ,0}, μ={μ,1}. Same contraction → K_μ.
//   5. ρ path: ü·λ^* — scalar, no AD.
//
// Pseudo-Hessian uses the same decomposition with forward contractions.

template <int NGLL>
void IsotropicElasticSensitivityAD2D::AccumulateKernel(
    const Vector& fwd_u, const Vector& fwd_a,
    const Vector& adj_u, real_t dt)
{
    const int ne = ne_;

    const real_t* d_fwd_u = fwd_u.Read();
    const real_t* d_fwd_a = fwd_a.Read();
    const real_t* d_adj_u = adj_u.Read();

    real_t* d_k_lambda = k_lambda_.ReadWrite();
    real_t* d_k_mu     = k_mu_.ReadWrite();
    real_t* d_k_rho    = k_rho_.ReadWrite();
    real_t* d_h_lambda = h_lambda_.ReadWrite();
    real_t* d_h_mu     = h_mu_.ReadWrite();
    real_t* d_h_rho    = h_rho_.ReadWrite();

    auto gmap_x = dofs_.ViewGatherMapX();
    auto gmap_y = dofs_.ViewGatherMapY();
    auto invJP  = geom_.ViewInvJPacked();
    auto dshape = geom_.ViewDShape();
    auto lambda_view = Reshape(lambda_.Read(), ngll_, ngll_, ne_);
    auto mu_view     = Reshape(mu_.Read(),     ngll_, ngll_, ne_);

    mfem::forall_2D(ne, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei)
    {
        MFEM_SHARED real_t s_ux[NGLL][NGLL];
        MFEM_SHARED real_t s_uy[NGLL][NGLL];
        MFEM_SHARED real_t s_lx[NGLL][NGLL];
        MFEM_SHARED real_t s_ly[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime[NGLL][NGLL];

        if (MFEM_THREAD_ID(y) == 0) {
            MFEM_FOREACH_THREAD(ix, x, NGLL) {
                for (int k = 0; k < NGLL; ++k) s_hprime[ix][k] = dshape(k, ix);
            }
        }

        MFEM_FOREACH_THREAD(iy, y, NGLL) {
            MFEM_FOREACH_THREAD(ix, x, NGLL) {
                const int gx = gmap_x(ix, iy, ei);
                const int gy = gmap_y(ix, iy, ei);
                s_ux[iy][ix] = d_fwd_u[gx];
                s_uy[iy][ix] = d_fwd_u[gy];
                s_lx[iy][ix] = d_adj_u[gx];
                s_ly[iy][ix] = d_adj_u[gy];
            }
        }
        MFEM_SYNC_THREAD;

        MFEM_FOREACH_THREAD(iy, y, NGLL) {
            MFEM_FOREACH_THREAD(ix, x, NGLL) {
                // Reference gradients.
                real_t dux_xi = 0, dux_eta = 0;
                real_t duy_xi = 0, duy_eta = 0;
                real_t dlx_xi = 0, dlx_eta = 0;
                real_t dly_xi = 0, dly_eta = 0;

                SEM_NOUNROLL
                for (int k = 0; k < NGLL; ++k) {
                    const real_t hp_x = s_hprime[ix][k];
                    dux_xi += s_ux[iy][k] * hp_x;
                    duy_xi += s_uy[iy][k] * hp_x;
                    dlx_xi += s_lx[iy][k] * hp_x;
                    dly_xi += s_ly[iy][k] * hp_x;

                    const real_t hp_y = s_hprime[iy][k];
                    dux_eta += s_ux[k][ix] * hp_y;
                    duy_eta += s_uy[k][ix] * hp_y;
                    dlx_eta += s_lx[k][ix] * hp_y;
                    dly_eta += s_ly[k][ix] * hp_y;
                }

                // Physical gradients via the packed inverse Jacobian.
                const real_t j0 = invJP(0, ix, iy, ei);
                const real_t j1 = invJP(1, ix, iy, ei);
                const real_t j2 = invJP(2, ix, iy, ei);
                const real_t j3 = invJP(3, ix, iy, ei);

                // See IsotropicElasticSensitivity2D.cpp for the nvcc
                // restriction this works around (no nested extended
                // __host__ __device__ lambdas, no reference captures).
                auto to_phys = [=]
                    (real_t d_xi, real_t d_eta, real_t& d_x, real_t& d_y) {
                        d_x = d_xi * j0 + d_eta * j3;
                        d_y = d_xi * j2 + d_eta * j1;
                    };

                real_t dux_dx, dux_dy, duy_dx, duy_dy;
                real_t dlx_dx, dlx_dy, dly_dx, dly_dy;
                to_phys(dux_xi, dux_eta, dux_dx, dux_dy);
                to_phys(duy_xi, duy_eta, duy_dx, duy_dy);
                to_phys(dlx_xi, dlx_eta, dlx_dx, dlx_dy);
                to_phys(dly_xi, dly_eta, dly_dx, dly_dy);

                // Adjoint strain (symmetric ε(λ^*)) for the stress:strain contraction.
                const real_t eps_adj_xx = dlx_dx;
                const real_t eps_adj_yy = dly_dy;
                const real_t eps_adj_xy = 0.5 * (dlx_dy + dly_dx);

                // Forward strain for the pseudo-Hessian.
                const real_t eps_fwd_xx = dux_dx;
                const real_t eps_fwd_yy = duy_dy;
                const real_t eps_fwd_xy = 0.5 * (dux_dy + duy_dx);

                // No MFEM_HOST_DEVICE: nested inside forall_2D's lambda
                // (see IsotropicElasticSensitivity2D.cpp:to_phys for the
                // same rationale).
                auto sigma_colon_eps = [] (
                    const Dual& sxx, const Dual& sxy, const Dual& syy,
                    real_t exx, real_t exy, real_t eyy) -> Dual {
                        return sxx * exx + syy * eyy + (sxy + sxy) * exy;
                    };

                const real_t lam_val = lambda_view(ix, iy, ei);
                const real_t mu_val  = mu_view(ix, iy, ei);

                // --- AD pass 1: seed λ ---------------------------------------
                {
                    const Dual lam_d{lam_val, 1.0};
                    const Dual mu_d {mu_val,  0.0};
                    Dual sxx, sxy, syy;
                    ElasticStressPhysical2D(lam_d, mu_d,
                                            dux_dx, dux_dy, duy_dx, duy_dy,
                                            sxx, sxy, syy);
                    // σ:ε(λ^*) is dual; .gradient is ε(λ^*):∂σ/∂λ = K_λ integrand.
                    const Dual contract = sigma_colon_eps(sxx, sxy, syy,
                                                          eps_adj_xx,
                                                          eps_adj_xy,
                                                          eps_adj_yy);
                    // K_λ integrand — sign matches the classic kernel K_λ = -∫...dt.
                    const int loc = ix + iy * NGLL + ei * NGLL * NGLL;
                    d_k_lambda[loc] -= contract.gradient * dt;

                    // Pseudo-Hessian via forward-on-forward contraction.
                    const Dual h_contract = sigma_colon_eps(sxx, sxy, syy,
                                                            eps_fwd_xx,
                                                            eps_fwd_xy,
                                                            eps_fwd_yy);
                    d_h_lambda[loc] += h_contract.gradient * dt;
                }

                // --- AD pass 2: seed μ ---------------------------------------
                {
                    const Dual lam_d{lam_val, 0.0};
                    const Dual mu_d {mu_val,  1.0};
                    Dual sxx, sxy, syy;
                    ElasticStressPhysical2D(lam_d, mu_d,
                                            dux_dx, dux_dy, duy_dx, duy_dy,
                                            sxx, sxy, syy);
                    const Dual contract = sigma_colon_eps(sxx, sxy, syy,
                                                          eps_adj_xx,
                                                          eps_adj_xy,
                                                          eps_adj_yy);
                    const int loc = ix + iy * NGLL + ei * NGLL * NGLL;
                    d_k_mu[loc] -= contract.gradient * dt;

                    const Dual h_contract = sigma_colon_eps(sxx, sxy, syy,
                                                            eps_fwd_xx,
                                                            eps_fwd_xy,
                                                            eps_fwd_yy);
                    d_h_mu[loc] += h_contract.gradient * dt;
                }

                // --- ρ path (direct, no AD needed) --------------------------
                {
                    const int gx = gmap_x(ix, iy, ei);
                    const int gy = gmap_y(ix, iy, ei);
                    const real_t ax = d_fwd_a[gx];
                    const real_t ay = d_fwd_a[gy];
                    const real_t lx = d_adj_u[gx];
                    const real_t ly = d_adj_u[gy];

                    const int loc = ix + iy * NGLL + ei * NGLL * NGLL;
                    d_k_rho[loc] -= (ax * lx + ay * ly) * dt;
                    d_h_rho[loc] += (ax * ax + ay * ay) * dt;
                }
            }
        }
    });
}

// =============================================================================
// Save — chain rule (λ, μ, ρ) → (Vp, Vs, ρ).
// =============================================================================

void IsotropicElasticSensitivityAD2D::Save(
    const std::string& dir, ParMesh& /*mesh*/, int source_id)
{
    const int total = ngll_ * ngll_ * ne_;

    MaterialField vp_field(ne_, ngll_, ngll_);
    MaterialField vs_field(ne_, ngll_, ngll_);
    MaterialField rho_field(ne_, ngll_, ngll_);

    const real_t* h_klam  = k_lambda_.HostRead();
    const real_t* h_kmu   = k_mu_.HostRead();
    const real_t* h_krho  = k_rho_.HostRead();
    const real_t* h_cvp   = coeff_vp_.HostRead();
    const real_t* h_cvsmu = coeff_vs_mu_.HostRead();
    const real_t* h_cvslm = coeff_vs_lam_.HostRead();

    real_t* h_vp  = vp_field.HostWrite();
    real_t* h_vs  = vs_field.HostWrite();
    real_t* h_rho = rho_field.HostWrite();
    for (int i = 0; i < total; ++i) {
        const real_t klam = h_klam[i];
        const real_t kmu  = h_kmu[i];
        h_vp[i]  = h_cvp[i] * klam;
        h_vs[i]  = h_cvsmu[i] * kmu + h_cvslm[i] * klam;
        h_rho[i] = h_krho[i];
    }

    std::ostringstream oss;
    oss << "_src" << std::setfill('0') << std::setw(3) << source_id;
    const std::string suffix = oss.str();
    const std::string mesh_path = dir + "/mesh.mesh";

    MPI_Comm comm = fes_->GetComm();
    SaveFieldBP(dir + "/kernel_vp"  + suffix + ".bp", "data", vp_field,  mesh_path, comm);
    SaveFieldBP(dir + "/kernel_vs"  + suffix + ".bp", "data", vs_field,  mesh_path, comm);
    SaveFieldBP(dir + "/kernel_rho" + suffix + ".bp", "data", rho_field, mesh_path, comm);
}

void IsotropicElasticSensitivityAD2D::SaveHessian(
    const std::string& dir, ParMesh& /*mesh*/, int source_id)
{
    const int total = ngll_ * ngll_ * ne_;

    MaterialField hvp_field(ne_, ngll_, ngll_);
    MaterialField hvs_field(ne_, ngll_, ngll_);
    MaterialField hrho_field(ne_, ngll_, ngll_);

    const real_t* h_hlam  = h_lambda_.HostRead();
    const real_t* h_hmu   = h_mu_.HostRead();
    const real_t* h_hrho  = h_rho_.HostRead();
    const real_t* h_cvp   = coeff_vp_.HostRead();
    const real_t* h_cvsmu = coeff_vs_mu_.HostRead();
    const real_t* h_cvslm = coeff_vs_lam_.HostRead();

    real_t* h_vp  = hvp_field.HostWrite();
    real_t* h_vs  = hvs_field.HostWrite();
    real_t* h_rho = hrho_field.HostWrite();
    for (int i = 0; i < total; ++i) {
        const real_t hlam = h_hlam[i];
        const real_t hmu  = h_hmu[i];
        h_vp[i]  = h_cvp[i] * h_cvp[i] * hlam;
        h_vs[i]  = h_cvsmu[i] * h_cvsmu[i] * hmu + h_cvslm[i] * h_cvslm[i] * hlam;
        h_rho[i] = h_hrho[i];
    }

    std::ostringstream oss;
    oss << "_src" << std::setfill('0') << std::setw(3) << source_id;
    const std::string suffix = oss.str();
    const std::string mesh_path = dir + "/mesh.mesh";

    MPI_Comm comm = fes_->GetComm();
    SaveFieldBP(dir + "/hessian_vp"  + suffix + ".bp", "data", hvp_field,  mesh_path, comm);
    SaveFieldBP(dir + "/hessian_vs"  + suffix + ".bp", "data", hvs_field,  mesh_path, comm);
    SaveFieldBP(dir + "/hessian_rho" + suffix + ".bp", "data", hrho_field, mesh_path, comm);
}

// =============================================================================
// NGLL instantiations
// =============================================================================

#define INSTANTIATE_ELASTIC_AD_2D(NGLL) \
    template void IsotropicElasticSensitivityAD2D::AccumulateKernel<NGLL>( \
        const Vector&, const Vector&, const Vector&, real_t);

SEM_INSTANTIATE_NGLL_2D(INSTANTIATE_ELASTIC_AD_2D)

#undef INSTANTIATE_ELASTIC_AD_2D

}  // namespace SEM
