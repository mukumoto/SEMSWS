/**
 * @file IsotropicElasticSensitivityAD3D.cpp
 * @brief AD sensitivity kernel implementation for 3D isotropic elastic.
 *
 * 3D analogue of IsotropicElasticSensitivityAD2D: seeds λ and μ separately
 * as `mfem::future::dual<real_t, real_t>`, invokes ElasticStressPhysical3D<T>,
 * and contracts the tangent-stress tensor with the adjoint strain using the
 * full 3D σ:ε formula
 *   σ:ε = σxx εxx + σyy εyy + σzz εzz + 2 (σxy εxy + σxz εxz + σyz εyz).
 * The ρ path (ü·λ^*) is pointwise, no dual needed. Save() matches the hand
 * version's TOY2DAC chain rule exactly.
 */

#include "fwi/IsotropicElasticSensitivityAD3D.hpp"
#include "integ/kernels/ElasticFluxKernel3D.hpp"
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

IsotropicElasticSensitivityAD3D::IsotropicElasticSensitivityAD3D(
    ParFiniteElementSpace& fes,
    const MaterialField3D& lambda,
    const MaterialField3D& mu,
    const MaterialField3D& rho,
    const MaterialField3D* c_kappa,
    const MaterialField3D* c_mu)
    : fes_(&fes)
{
    const int order = SafeFESOrder(fes);
    ngll_ = order + 1;
    ne_   = fes.GetParMesh()->GetNE();

    const int total_gll = ngll_ * ngll_ * ngll_ * ne_;

    geom_.Compute(fes);
    dofs_.ComputeVector(fes, order);
    geom_.EnableDevice();
    geom_.SyncToDevice();
    dofs_.EnableDevice();
    dofs_.SyncToDevice();

    auto alloc = [&](Vector& v) { v.SetSize(total_gll); v.UseDevice(true); };
    alloc(lambda_); alloc(mu_);
    alloc(coeff_vp_); alloc(coeff_vs_mu_); alloc(coeff_vs_lam_);
    InitCoefficients(lambda, mu, rho, c_kappa, c_mu);

    alloc(k_lambda_); k_lambda_ = 0.0;
    alloc(k_mu_);     k_mu_     = 0.0;
    alloc(k_rho_);    k_rho_    = 0.0;
    alloc(h_lambda_); h_lambda_ = 0.0;
    alloc(h_mu_);     h_mu_     = 0.0;
    alloc(h_rho_);    h_rho_    = 0.0;
}

void IsotropicElasticSensitivityAD3D::InitCoefficients(
    const MaterialField3D& lambda,
    const MaterialField3D& mu,
    const MaterialField3D& rho,
    const MaterialField3D* c_kappa,
    const MaterialField3D* c_mu)
{
    const int total_gll = ngll_ * ngll_ * ngll_ * ne_;
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
        const real_t lam_u = d_lam[i];
        const real_t mu_u  = d_mu[i];
        const real_t r     = d_rho[i];
        const real_t ck    = d_ck ? d_ck[i] : 1.0;
        const real_t cm    = d_cm ? d_cm[i] : 1.0;
        const real_t mu_user  = mu_u / cm;
        const real_t kappa_u  = lam_u + (2.0 / 3.0) * mu_u;
        const real_t lam_user = kappa_u / ck - (2.0 / 3.0) * mu_user;
        const real_t vp = sqrt((lam_user + 2.0 * mu_user) / r);
        const real_t vs = sqrt(mu_user / r);
        d_lambda_out[i]    = lam_u;
        d_mu_out[i]        = mu_u;
        // Pure elastic form of chain rule — see comment in
        // IsotropicElasticSensitivity2D.cpp / ...3D.cpp for why c_κ / c_μ
        // must NOT multiply the Save-time coefficients for SLS visco-elastic.
        (void)ck; (void)cm;
        d_coeff_vp[i]      = 2.0 * r * vp;
        d_coeff_vs_mu[i]   = 2.0 * r * vs;
        d_coeff_vs_lam[i]  = -4.0 * r * vs;
    });
}

// =============================================================================
// Reset
// =============================================================================

void IsotropicElasticSensitivityAD3D::Reset() {
    k_lambda_ = 0.0; k_mu_ = 0.0; k_rho_ = 0.0;
}

void IsotropicElasticSensitivityAD3D::ResetHessian() {
    h_lambda_ = 0.0; h_mu_ = 0.0; h_rho_ = 0.0;
}

// =============================================================================
// Accumulate (NGLL dispatch)
// =============================================================================

void IsotropicElasticSensitivityAD3D::Accumulate(
    const Vector& fwd_u, const Vector& fwd_a,
    const Vector& adj_u, real_t dt)
{
    SEM_DISPATCH_NGLL_3D(ngll_, AccumulateKernel, fwd_u, fwd_a, adj_u, dt);
}

// =============================================================================
// AccumulateKernel<NGLL>
// =============================================================================

template <int NGLL>
void IsotropicElasticSensitivityAD3D::AccumulateKernel(
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
    auto gmap_z = dofs_.ViewGatherMapZ();
    auto invJP  = geom_.ViewInvJPacked();
    auto dshape = geom_.ViewDShape();
    auto lambda_view = Reshape(lambda_.Read(), ngll_, ngll_, ngll_, ne_);
    auto mu_view     = Reshape(mu_.Read(),     ngll_, ngll_, ngll_, ne_);

    mfem::forall_3D(ne, NGLL, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei)
    {
        MFEM_SHARED real_t s_ux[NGLL][NGLL][NGLL];
        MFEM_SHARED real_t s_uy[NGLL][NGLL][NGLL];
        MFEM_SHARED real_t s_uz[NGLL][NGLL][NGLL];
        MFEM_SHARED real_t s_lx[NGLL][NGLL][NGLL];
        MFEM_SHARED real_t s_ly[NGLL][NGLL][NGLL];
        MFEM_SHARED real_t s_lz[NGLL][NGLL][NGLL];
        MFEM_SHARED real_t s_hprime[NGLL][NGLL];

        if (MFEM_THREAD_ID(z) == 0) {
            MFEM_FOREACH_THREAD(iy, y, NGLL) {
                MFEM_FOREACH_THREAD(ix, x, NGLL) {
                    s_hprime[ix][iy] = dshape(iy, ix);
                }
            }
        }

        MFEM_FOREACH_THREAD(iz, z, NGLL) {
            MFEM_FOREACH_THREAD(iy, y, NGLL) {
                MFEM_FOREACH_THREAD(ix, x, NGLL) {
                    const int gx = gmap_x(ix, iy, iz, ei);
                    const int gy = gmap_y(ix, iy, iz, ei);
                    const int gz = gmap_z(ix, iy, iz, ei);
                    s_ux[iz][iy][ix] = d_fwd_u[gx];
                    s_uy[iz][iy][ix] = d_fwd_u[gy];
                    s_uz[iz][iy][ix] = d_fwd_u[gz];
                    s_lx[iz][iy][ix] = d_adj_u[gx];
                    s_ly[iz][iy][ix] = d_adj_u[gy];
                    s_lz[iz][iy][ix] = d_adj_u[gz];
                }
            }
        }
        MFEM_SYNC_THREAD;

        MFEM_FOREACH_THREAD(iz, z, NGLL) {
            MFEM_FOREACH_THREAD(iy, y, NGLL) {
                MFEM_FOREACH_THREAD(ix, x, NGLL) {
                    real_t dux_xi = 0, dux_eta = 0, dux_zeta = 0;
                    real_t duy_xi = 0, duy_eta = 0, duy_zeta = 0;
                    real_t duz_xi = 0, duz_eta = 0, duz_zeta = 0;
                    real_t dlx_xi = 0, dlx_eta = 0, dlx_zeta = 0;
                    real_t dly_xi = 0, dly_eta = 0, dly_zeta = 0;
                    real_t dlz_xi = 0, dlz_eta = 0, dlz_zeta = 0;

                    SEM_NOUNROLL
                    for (int k = 0; k < NGLL; ++k) {
                        const real_t hp_x = s_hprime[ix][k];
                        dux_xi += s_ux[iz][iy][k] * hp_x;
                        duy_xi += s_uy[iz][iy][k] * hp_x;
                        duz_xi += s_uz[iz][iy][k] * hp_x;
                        dlx_xi += s_lx[iz][iy][k] * hp_x;
                        dly_xi += s_ly[iz][iy][k] * hp_x;
                        dlz_xi += s_lz[iz][iy][k] * hp_x;

                        const real_t hp_y = s_hprime[iy][k];
                        dux_eta += s_ux[iz][k][ix] * hp_y;
                        duy_eta += s_uy[iz][k][ix] * hp_y;
                        duz_eta += s_uz[iz][k][ix] * hp_y;
                        dlx_eta += s_lx[iz][k][ix] * hp_y;
                        dly_eta += s_ly[iz][k][ix] * hp_y;
                        dlz_eta += s_lz[iz][k][ix] * hp_y;

                        const real_t hp_z = s_hprime[iz][k];
                        dux_zeta += s_ux[k][iy][ix] * hp_z;
                        duy_zeta += s_uy[k][iy][ix] * hp_z;
                        duz_zeta += s_uz[k][iy][ix] * hp_z;
                        dlx_zeta += s_lx[k][iy][ix] * hp_z;
                        dly_zeta += s_ly[k][iy][ix] * hp_z;
                        dlz_zeta += s_lz[k][iy][ix] * hp_z;
                    }

                    const real_t j0 = invJP(0, ix, iy, iz, ei);
                    const real_t j1 = invJP(1, ix, iy, iz, ei);
                    const real_t j2 = invJP(2, ix, iy, iz, ei);
                    const real_t j3 = invJP(3, ix, iy, iz, ei);
                    const real_t j4 = invJP(4, ix, iy, iz, ei);
                    const real_t j5 = invJP(5, ix, iy, iz, ei);
                    const real_t j6 = invJP(6, ix, iy, iz, ei);
                    const real_t j7 = invJP(7, ix, iy, iz, ei);
                    const real_t j8 = invJP(8, ix, iy, iz, ei);

                    // See IsotropicElasticSensitivity2D.cpp for the nvcc
                    // restriction this works around (no nested extended
                    // __host__ __device__ lambdas, no reference captures).
                    auto to_phys = [=]
                        (real_t dxi, real_t deta, real_t dzeta,
                         real_t& dx, real_t& dy, real_t& dz) {
                            dx = dxi * j0 + deta * j3 + dzeta * j6;
                            dy = dxi * j1 + deta * j4 + dzeta * j7;
                            dz = dxi * j2 + deta * j5 + dzeta * j8;
                        };

                    real_t dux_dx, dux_dy, dux_dz;
                    real_t duy_dx, duy_dy, duy_dz;
                    real_t duz_dx, duz_dy, duz_dz;
                    real_t dlx_dx, dlx_dy, dlx_dz;
                    real_t dly_dx, dly_dy, dly_dz;
                    real_t dlz_dx, dlz_dy, dlz_dz;
                    to_phys(dux_xi, dux_eta, dux_zeta, dux_dx, dux_dy, dux_dz);
                    to_phys(duy_xi, duy_eta, duy_zeta, duy_dx, duy_dy, duy_dz);
                    to_phys(duz_xi, duz_eta, duz_zeta, duz_dx, duz_dy, duz_dz);
                    to_phys(dlx_xi, dlx_eta, dlx_zeta, dlx_dx, dlx_dy, dlx_dz);
                    to_phys(dly_xi, dly_eta, dly_zeta, dly_dx, dly_dy, dly_dz);
                    to_phys(dlz_xi, dlz_eta, dlz_zeta, dlz_dx, dlz_dy, dlz_dz);

                    // Symmetric adjoint strain ε(λ^*).
                    const real_t eps_adj_xx = dlx_dx;
                    const real_t eps_adj_yy = dly_dy;
                    const real_t eps_adj_zz = dlz_dz;
                    const real_t eps_adj_xy = 0.5 * (dlx_dy + dly_dx);
                    const real_t eps_adj_xz = 0.5 * (dlx_dz + dlz_dx);
                    const real_t eps_adj_yz = 0.5 * (dly_dz + dlz_dy);

                    // Symmetric forward strain (for pseudo-Hessian).
                    const real_t eps_fwd_xx = dux_dx;
                    const real_t eps_fwd_yy = duy_dy;
                    const real_t eps_fwd_zz = duz_dz;
                    const real_t eps_fwd_xy = 0.5 * (dux_dy + duy_dx);
                    const real_t eps_fwd_xz = 0.5 * (dux_dz + duz_dx);
                    const real_t eps_fwd_yz = 0.5 * (duy_dz + duz_dy);

                    // No MFEM_HOST_DEVICE: nested inside forall_3D's lambda
                    // (see IsotropicElasticSensitivity2D.cpp:to_phys for
                    // the same rationale).
                    auto sigma_colon_eps = [] (
                        const Dual& sxx, const Dual& syy, const Dual& szz,
                        const Dual& sxy, const Dual& sxz, const Dual& syz,
                        real_t exx, real_t eyy, real_t ezz,
                        real_t exy, real_t exz, real_t eyz) -> Dual {
                            return sxx * exx + syy * eyy + szz * ezz
                                 + (sxy + sxy) * exy
                                 + (sxz + sxz) * exz
                                 + (syz + syz) * eyz;
                        };

                    const real_t lam_val = lambda_view(ix, iy, iz, ei);
                    const real_t mu_val  = mu_view(ix, iy, iz, ei);

                    const int loc =
                        ix + iy * NGLL + iz * NGLL * NGLL
                        + ei * NGLL * NGLL * NGLL;

                    // --- AD pass 1: seed λ -----------------------------------
                    {
                        const Dual lam_d{lam_val, 1.0};
                        const Dual mu_d {mu_val,  0.0};
                        Dual sxx, syy, szz, sxy, sxz, syz;
                        ElasticStressPhysical3D(lam_d, mu_d,
                            dux_dx, dux_dy, dux_dz,
                            duy_dx, duy_dy, duy_dz,
                            duz_dx, duz_dy, duz_dz,
                            sxx, syy, szz, sxy, sxz, syz);
                        const Dual contract = sigma_colon_eps(
                            sxx, syy, szz, sxy, sxz, syz,
                            eps_adj_xx, eps_adj_yy, eps_adj_zz,
                            eps_adj_xy, eps_adj_xz, eps_adj_yz);
                        d_k_lambda[loc] -= contract.gradient * dt;

                        const Dual h_contract = sigma_colon_eps(
                            sxx, syy, szz, sxy, sxz, syz,
                            eps_fwd_xx, eps_fwd_yy, eps_fwd_zz,
                            eps_fwd_xy, eps_fwd_xz, eps_fwd_yz);
                        d_h_lambda[loc] += h_contract.gradient * dt;
                    }

                    // --- AD pass 2: seed μ -----------------------------------
                    {
                        const Dual lam_d{lam_val, 0.0};
                        const Dual mu_d {mu_val,  1.0};
                        Dual sxx, syy, szz, sxy, sxz, syz;
                        ElasticStressPhysical3D(lam_d, mu_d,
                            dux_dx, dux_dy, dux_dz,
                            duy_dx, duy_dy, duy_dz,
                            duz_dx, duz_dy, duz_dz,
                            sxx, syy, szz, sxy, sxz, syz);
                        const Dual contract = sigma_colon_eps(
                            sxx, syy, szz, sxy, sxz, syz,
                            eps_adj_xx, eps_adj_yy, eps_adj_zz,
                            eps_adj_xy, eps_adj_xz, eps_adj_yz);
                        d_k_mu[loc] -= contract.gradient * dt;

                        const Dual h_contract = sigma_colon_eps(
                            sxx, syy, szz, sxy, sxz, syz,
                            eps_fwd_xx, eps_fwd_yy, eps_fwd_zz,
                            eps_fwd_xy, eps_fwd_xz, eps_fwd_yz);
                        d_h_mu[loc] += h_contract.gradient * dt;
                    }

                    // --- ρ path (direct, no AD) -----------------------------
                    {
                        const int gx = gmap_x(ix, iy, iz, ei);
                        const int gy = gmap_y(ix, iy, iz, ei);
                        const int gz = gmap_z(ix, iy, iz, ei);
                        const real_t ax = d_fwd_a[gx];
                        const real_t ay = d_fwd_a[gy];
                        const real_t az = d_fwd_a[gz];
                        const real_t lx = d_adj_u[gx];
                        const real_t ly = d_adj_u[gy];
                        const real_t lz = d_adj_u[gz];
                        d_k_rho[loc] -= (ax * lx + ay * ly + az * lz) * dt;
                        d_h_rho[loc] += (ax * ax + ay * ay + az * az) * dt;
                    }
                }
            }
        }
    });
}

// =============================================================================
// Save — same TOY2DAC chain rule as hand 3D
// =============================================================================

void IsotropicElasticSensitivityAD3D::Save(
    const std::string& dir, ParMesh& /*mesh*/, int source_id)
{
    const int total = ngll_ * ngll_ * ngll_ * ne_;

    MaterialField3D vp_field(ne_, ngll_, ngll_, ngll_);
    MaterialField3D vs_field(ne_, ngll_, ngll_, ngll_);
    MaterialField3D rho_field(ne_, ngll_, ngll_, ngll_);

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
        h_vp[i]  = h_cvp[i] * h_klam[i];
        h_vs[i]  = h_cvsmu[i] * h_kmu[i] + h_cvslm[i] * h_klam[i];
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

void IsotropicElasticSensitivityAD3D::SaveHessian(
    const std::string& dir, ParMesh& /*mesh*/, int source_id)
{
    const int total = ngll_ * ngll_ * ngll_ * ne_;

    MaterialField3D hvp(ne_, ngll_, ngll_, ngll_);
    MaterialField3D hvs(ne_, ngll_, ngll_, ngll_);
    MaterialField3D hrho(ne_, ngll_, ngll_, ngll_);

    const real_t* h_hlam  = h_lambda_.HostRead();
    const real_t* h_hmu   = h_mu_.HostRead();
    const real_t* h_hrho  = h_rho_.HostRead();
    const real_t* h_cvp   = coeff_vp_.HostRead();
    const real_t* h_cvsmu = coeff_vs_mu_.HostRead();
    const real_t* h_cvslm = coeff_vs_lam_.HostRead();

    real_t* h_vp  = hvp.HostWrite();
    real_t* h_vs  = hvs.HostWrite();
    real_t* h_rho = hrho.HostWrite();
    for (int i = 0; i < total; ++i) {
        h_vp[i]  = h_cvp[i] * h_cvp[i] * h_hlam[i];
        h_vs[i]  = h_cvsmu[i] * h_cvsmu[i] * h_hmu[i]
                 + h_cvslm[i] * h_cvslm[i] * h_hlam[i];
        h_rho[i] = h_hrho[i];
    }

    std::ostringstream oss;
    oss << "_src" << std::setfill('0') << std::setw(3) << source_id;
    const std::string suffix = oss.str();
    const std::string mesh_path = dir + "/mesh.mesh";

    MPI_Comm comm = fes_->GetComm();
    SaveFieldBP(dir + "/hessian_vp"  + suffix + ".bp", "data", hvp,  mesh_path, comm);
    SaveFieldBP(dir + "/hessian_vs"  + suffix + ".bp", "data", hvs,  mesh_path, comm);
    SaveFieldBP(dir + "/hessian_rho" + suffix + ".bp", "data", hrho, mesh_path, comm);
}

// =============================================================================
// NGLL instantiations
// =============================================================================

#define INSTANTIATE_ELASTIC_AD_3D(NGLL) \
    template void IsotropicElasticSensitivityAD3D::AccumulateKernel<NGLL>( \
        const Vector&, const Vector&, const Vector&, real_t);

SEM_INSTANTIATE_NGLL_3D(INSTANTIATE_ELASTIC_AD_3D)

#undef INSTANTIATE_ELASTIC_AD_3D

}  // namespace SEM
