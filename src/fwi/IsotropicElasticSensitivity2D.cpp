/**
 * @file IsotropicElasticSensitivity2D.cpp
 * @brief Hand-written sensitivity kernel for 2D isotropic elastic media.
 *
 * See the header for the math. Implementation follows the acoustic hand
 * version (4-phase tensor-product kernel with shared memory, one
 * `AccumulateKernel<NGLL>` doing everything), but with vector (u_x, u_y)
 * gathers, the divergence + full strain contractions, and a scalar
 * ρ-path (ü · λ^*) instead of the Laplacian form.
 *
 * Save() applies the chain rule (λ, μ, ρ) → (Vp, Vs, ρ) so the BP output
 * is directly consumable by downstream tools (sem_viz, etc).
 */

#include "fwi/IsotropicElasticSensitivity2D.hpp"
#include "util/FESOrder.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "common/GpuMacros.hpp"
#include <mfem.hpp>
#include "io/ADIOS2IO.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace SEM {

// =============================================================================
// Constructor
// =============================================================================

IsotropicElasticSensitivity2D::IsotropicElasticSensitivity2D(
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

    // Geometry + DOF ordering — vector version, exposes both
    // gather_map_x and gather_map_y on the vdim=2 space.
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

    alloc_dev(rho_);
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

// =============================================================================
// InitCoefficients
// =============================================================================

void IsotropicElasticSensitivity2D::InitCoefficients(
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
    real_t* d_rho_out       = rho_.Write();
    real_t* d_coeff_vp      = coeff_vp_.Write();
    real_t* d_coeff_vs_mu   = coeff_vs_mu_.Write();
    real_t* d_coeff_vs_lam  = coeff_vs_lam_.Write();

    mfem::forall(total_gll, [=] MFEM_HOST_DEVICE (int i) {
        const real_t lam_u = d_lam[i];          // unrelaxed if viscoelastic
        const real_t mu_u  = d_mu[i];
        const real_t r     = d_rho[i];
        const real_t ck    = d_ck ? d_ck[i] : 1.0;
        const real_t cm    = d_cm ? d_cm[i] : 1.0;
        // User-facing moduli (inverse of the unrelaxed correction).
        const real_t mu_user  = mu_u / cm;
        // Plane-strain 2D: κ = λ + μ. λ_user = κ_user − μ_user = κ_u/c_κ − μ_u/c_μ.
        const real_t kappa_u  = lam_u + mu_u;
        const real_t lam_user = kappa_u / ck - mu_user;
        const real_t vp = sqrt((lam_user + 2.0 * mu_user) / r);
        const real_t vs = sqrt(mu_user / r);
        d_rho_out[i]       = r;
        // Save-time chain rule — SLS visco-elastic uses the PURE ELASTIC form.
        //
        // Naively one would chain through the stored-modulus perturbation:
        //   dJ/dVp_user = dJ/dλ_u · dλ_u/dVp_user = K_λu · 2·c_κ·ρ·Vp
        // but this double-counts the unrelaxed correction. The SLS memory
        // variables enforce that the wavefield's EFFECTIVE modulus at the
        // reference frequency f0 equals the user-facing κ_user (by Q-fit
        // construction), so the accumulated `(∇·u_fwd)(∇·u_adj)·dt` kernel
        // is physically the sensitivity to λ_user, not λ_u. Chain rule:
        //   K_Vp = 2·ρ·Vp_user   · K_λ
        //   K_Vs = 2·ρ·Vs_user   · K_μ − 4·ρ·Vs_user · K_λ    (pure elastic)
        // c_κ, c_μ still appear when reconstructing Vp_user, Vs_user from
        // the stored (unrelaxed) moduli above, but they must NOT multiply
        // the Save-time coefficients. Validated numerically: 2D elastic
        // slope=1 at Q=9999, Q=20/20, and Q=30/20 cases.
        // (Pure elastic: ck==cm==1, so the old and new formulas coincide.)
        (void)ck; (void)cm;
        d_coeff_vp[i]      = 2.0 * r * vp;
        d_coeff_vs_mu[i]   = 2.0 * r * vs;
        d_coeff_vs_lam[i]  = -4.0 * r * vs;
    });
}

// =============================================================================
// Reset / ResetHessian
// =============================================================================

void IsotropicElasticSensitivity2D::Reset() {
    k_lambda_ = 0.0;
    k_mu_     = 0.0;
    k_rho_    = 0.0;
}

void IsotropicElasticSensitivity2D::ResetHessian() {
    h_lambda_ = 0.0;
    h_mu_     = 0.0;
    h_rho_    = 0.0;
}

// =============================================================================
// Accumulate (dispatch to NGLL-specialized GPU kernel)
// =============================================================================

void IsotropicElasticSensitivity2D::Accumulate(
    const Vector& fwd_u, const Vector& fwd_a,
    const Vector& adj_u, real_t dt)
{
    SEM_DISPATCH_NGLL(ngll_, AccumulateKernel, fwd_u, fwd_a, adj_u, dt);
}

// =============================================================================
// AccumulateKernel<NGLL>
//
//   K_λ += -(∇·u)(∇·λ^*) · dt
//   K_μ += -2 ε(u):ε(λ^*) · dt
//   K_ρ += -(u_ẍ·λ^*_x + u_ÿ·λ^*_y) · dt
//
// Plus the pseudo-Hessian forward-illumination counterparts.
// =============================================================================

template <int NGLL>
void IsotropicElasticSensitivity2D::AccumulateKernel(
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

    mfem::forall_2D(ne, NGLL, NGLL, [=] MFEM_HOST_DEVICE (int ei)
    {
        // ---- Shared memory ---------------------------------------------------
        MFEM_SHARED real_t s_ux[NGLL][NGLL];
        MFEM_SHARED real_t s_uy[NGLL][NGLL];
        MFEM_SHARED real_t s_lx[NGLL][NGLL];
        MFEM_SHARED real_t s_ly[NGLL][NGLL];
        MFEM_SHARED real_t s_hprime[NGLL][NGLL];

        // ---- Phase 1: basis derivatives -------------------------------------
        if (MFEM_THREAD_ID(y) == 0) {
            MFEM_FOREACH_THREAD(ix, x, NGLL) {
                for (int k = 0; k < NGLL; ++k) s_hprime[ix][k] = dshape(k, ix);
            }
        }

        // ---- Phase 2: gather forward u and adjoint λ^* ----------------------
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

        // ---- Phase 3: compute grads, contractions, accumulate ---------------
        MFEM_FOREACH_THREAD(iy, y, NGLL) {
            MFEM_FOREACH_THREAD(ix, x, NGLL) {

                // Reference gradients of forward and adjoint components.
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

                // Inverse Jacobian entries (reference → physical).
                const real_t j0 = invJP(0, ix, iy, ei);
                const real_t j1 = invJP(1, ix, iy, ei);
                const real_t j2 = invJP(2, ix, iy, ei);
                const real_t j3 = invJP(3, ix, iy, ei);

                // No MFEM_HOST_DEVICE: nvcc forbids defining an extended
                // __host__ __device__ lambda inside another (we're already
                // inside forall_2D's lambda). The inner lambda still runs
                // on device because the surrounding context already is.
                // [=] instead of [&]: extended host/device lambdas cannot
                // capture by reference, and j0..j3 are const scalars.
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

                // Divergences.
                const real_t div_u = dux_dx + duy_dy;
                const real_t div_l = dlx_dx + dly_dy;

                // Strain–strain contraction ε(u):ε(λ^*).
                // ε_xx = ∂u_x/∂x, ε_yy = ∂u_y/∂y,
                // ε_xy = ½(∂u_x/∂y + ∂u_y/∂x).
                // ε:ε' = ε_xx·ε'_xx + 2 ε_xy·ε'_xy + ε_yy·ε'_yy.
                const real_t exy_u = 0.5 * (dux_dy + duy_dx);
                const real_t exy_l = 0.5 * (dlx_dy + dly_dx);
                const real_t eps_eps =
                    dux_dx * dlx_dx + duy_dy * dly_dy + 2.0 * exy_u * exy_l;

                // Forward self-contraction for pseudo-Hessian.
                const real_t exy_u_sq = exy_u * exy_u;
                const real_t eps_sq_u =
                    dux_dx * dux_dx + duy_dy * duy_dy + 2.0 * exy_u_sq;

                // Gradient (K_ρ) needs forward acceleration · adjoint u.
                const int gx = gmap_x(ix, iy, ei);
                const int gy = gmap_y(ix, iy, ei);
                const real_t ax = d_fwd_a[gx];
                const real_t ay = d_fwd_a[gy];
                const real_t lx = d_adj_u[gx];
                const real_t ly = d_adj_u[gy];

                const int loc = ix + iy * NGLL + ei * NGLL * NGLL;

                // Gradient kernels (note signs: -= ...).
                d_k_lambda[loc] -= div_u * div_l * dt;
                d_k_mu[loc]     -= 2.0 * eps_eps * dt;
                d_k_rho[loc]    -= (ax * lx + ay * ly) * dt;

                // Pseudo-Hessians (positive forward-illumination form).
                d_h_lambda[loc] += div_u * div_u * dt;
                d_h_mu[loc]     += 2.0 * eps_sq_u * dt;
                d_h_rho[loc]    += (ax * ax + ay * ay) * dt;
            }
        }
    });
}

// =============================================================================
// Save — chain rule (λ, μ, ρ) → (Vp, Vs, ρ)
// =============================================================================

namespace {

void copy_to_field(const Vector& src, MaterialField& dst) {
    const int n = src.Size();
    const real_t* s = src.HostRead();
    real_t* d = dst.HostWrite();
    for (int i = 0; i < n; ++i) d[i] = s[i];
}

}  // anonymous namespace

void IsotropicElasticSensitivity2D::Save(
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

void IsotropicElasticSensitivity2D::SaveHessian(
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
        // Sum of squared chain-rule coefficients (diagonal Gauss-Newton).
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
// Explicit NGLL instantiations
// =============================================================================

#define INSTANTIATE_ELASTIC_SENS_2D(NGLL) \
    template void IsotropicElasticSensitivity2D::AccumulateKernel<NGLL>( \
        const Vector&, const Vector&, const Vector&, real_t);

SEM_INSTANTIATE_NGLL_2D(INSTANTIATE_ELASTIC_SENS_2D)

#undef INSTANTIATE_ELASTIC_SENS_2D

}  // namespace SEM
