/**
 * @file HelmholtzSmoothing.cpp
 * @brief Helmholtz smoothing implementation
 *
 * Solve: (M + σ²K) u_s = M u
 * With normalize → smooth → restore for numerical stability.
 *
 * When restrict_attrs is non-empty, M and K are assembled only on
 * elements whose attribute is in the set. DOFs that belong exclusively
 * to non-restricted elements keep their original (input) values.
 */

#include "fwi/HelmholtzSmoothing.hpp"
#include <iostream>
#include <cmath>
#include <set>

namespace SEM {

// =============================================================================
// Helper: collect DOFs belonging to restricted elements
// =============================================================================

static std::set<int> GetRestrictedDofs(ParFiniteElementSpace& fes,
                                       const std::vector<int>& restrict_attrs) {
    std::set<int> dof_set;
    std::set<int> attr_set(restrict_attrs.begin(), restrict_attrs.end());
    ParMesh* pmesh = fes.GetParMesh();
    for (int i = 0; i < pmesh->GetNE(); i++) {
        if (attr_set.count(pmesh->GetAttribute(i))) {
            Array<int> dofs;
            fes.GetElementDofs(i, dofs);
            for (int j = 0; j < dofs.Size(); j++) {
                dof_set.insert(dofs[j] >= 0 ? dofs[j] : -1 - dofs[j]);
            }
        }
    }
    return dof_set;
}

// =============================================================================
// Helper: get non-restricted TRUE-DOF Array (for EliminateRowsCols)
// =============================================================================

static Array<int> GetNonRestrictedTrueDofs(ParFiniteElementSpace& fes,
                                            const std::vector<int>& restrict_attrs) {
    std::set<int> ldof_set = GetRestrictedDofs(fes, restrict_attrs);

    // Project a marker from local DOFs to true DOFs
    Vector marker_l(fes.GetVSize());
    marker_l = 0.0;
    for (int d : ldof_set) marker_l(d) = 1.0;

    Vector marker_t(fes.GetTrueVSize());
    marker_t = 0.0;
    const auto* R = fes.GetRestrictionMatrix();
    if (R) {
        R->Mult(marker_l, marker_t);
    } else {
        marker_t = marker_l;
    }

    // Collect true DOFs NOT in restricted set
    Array<int> non_restricted;
    for (int i = 0; i < marker_t.Size(); i++) {
        if (marker_t(i) < 0.5) {
            non_restricted.Append(i);
        }
    }
    return non_restricted;
}

// =============================================================================
// Coefficient classes
// =============================================================================

/// Coefficient class for σ(x)² = (α/f)² · Vp(x)²
class VpSquaredCoefficient : public Coefficient {
public:
    VpSquaredCoefficient(const ParGridFunction& vp_gf, real_t alpha_over_f)
        : vp_coeff_(&vp_gf),
          scale_(alpha_over_f * alpha_over_f) {}

    real_t Eval(ElementTransformation& T,
                const IntegrationPoint& ip) override {
        real_t vp = vp_coeff_.Eval(T, ip);
        return scale_ * vp * vp;
    }

private:
    GridFunctionCoefficient vp_coeff_;
    real_t scale_;
};

/// MatrixCoefficient for anisotropic variable σ: D = diag((α_x/f)²·Vp², (α_y/f)²·Vp²)
class AnisoVpSquaredCoefficient : public MatrixCoefficient {
public:
    AnisoVpSquaredCoefficient(const ParGridFunction& vp_gf,
                              real_t alpha_x_over_f, real_t alpha_y_over_f)
        : MatrixCoefficient(2),
          vp_coeff_(&vp_gf),
          scale_x_(alpha_x_over_f * alpha_x_over_f),
          scale_y_(alpha_y_over_f * alpha_y_over_f) {}

    void Eval(DenseMatrix& K, ElementTransformation& T,
              const IntegrationPoint& ip) override {
        real_t vp = vp_coeff_.Eval(T, ip);
        real_t vp2 = vp * vp;
        K.SetSize(2, 2);
        K = 0.0;
        K(0, 0) = scale_x_ * vp2;
        K(1, 1) = scale_y_ * vp2;
    }

private:
    GridFunctionCoefficient vp_coeff_;
    real_t scale_x_, scale_y_;
};

/// MatrixCoefficient for anisotropic constant σ: D = diag(σ_x², σ_y²)
class AnisoConstantCoefficient : public MatrixCoefficient {
public:
    AnisoConstantCoefficient(real_t sigma_x2, real_t sigma_y2)
        : MatrixCoefficient(2), sigma_x2_(sigma_x2), sigma_y2_(sigma_y2) {}

    void Eval(DenseMatrix& K, ElementTransformation& T,
              const IntegrationPoint& ip) override {
        K.SetSize(2, 2);
        K = 0.0;
        K(0, 0) = sigma_x2_;
        K(1, 1) = sigma_y2_;
    }

private:
    real_t sigma_x2_, sigma_y2_;
};

// =============================================================================
// Internal: isotropic solve (M + coeff*K) gf_smooth = M * gf
// =============================================================================

static void SmoothInternal(ParFiniteElementSpace& fes,
                           ParGridFunction& gf,
                           Coefficient& sigma2_coeff,
                           const std::vector<int>& restrict_attrs) {
    MPI_Comm comm = fes.GetComm();
    bool restricted = !restrict_attrs.empty();

    // Normalize: find global max absolute value
    real_t local_max = 0.0;
    for (int i = 0; i < gf.Size(); i++) {
        local_max = std::max(local_max, std::abs(gf(i)));
    }
    real_t global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);

    if (global_max == 0.0) return;  // skip zero field

    // Scale to [-1, 1]
    gf /= global_max;

    // Save original values for non-restricted DOFs
    Vector gf_orig;
    std::set<int> restricted_dofs;
    if (restricted) {
        gf_orig = gf;
        restricted_dofs = GetRestrictedDofs(fes, restrict_attrs);
    }

    // Assemble mass matrix M (full domain)
    ParBilinearForm mass(&fes);
    mass.AddDomainIntegrator(new MassIntegrator());
    mass.Assemble();
    mass.Finalize();

    // Assemble diffusion matrix K with σ² coefficient (full domain)
    ParBilinearForm diff(&fes);
    diff.AddDomainIntegrator(new DiffusionIntegrator(sigma2_coeff));
    diff.Assemble();
    diff.Finalize();

    // A = M + K_σ²
    std::unique_ptr<HypreParMatrix> M(mass.ParallelAssemble());
    std::unique_ptr<HypreParMatrix> K(diff.ParallelAssemble());
    std::unique_ptr<HypreParMatrix> A(Add(1.0, *M, 1.0, *K));

    // rhs = M * u
    HypreParVector rhs(comm, fes.GlobalTrueVSize(), fes.GetTrueDofOffsets());
    HypreParVector x(comm, fes.GlobalTrueVSize(), fes.GetTrueDofOffsets());
    gf.ParallelProject(x);
    M->Mult(x, rhs);

    // For restricted smoothing, impose Dirichlet-like conditions on
    // non-restricted DOFs: set A row/col to identity, adjust rhs to
    // preserve original values. Smoothing only affects restricted DOFs.
    if (restricted) {
        Array<int> nr_tdofs = GetNonRestrictedTrueDofs(fes, restrict_attrs);
        A->EliminateRowsCols(nr_tdofs, x, rhs);
    }

    // Solve with AMG-preconditioned CG
    x = 0.0;
    HypreBoomerAMG amg(*A);
    amg.SetPrintLevel(0);
    CGSolver cg(comm);
    cg.SetOperator(*A);
    cg.SetPreconditioner(amg);
    cg.SetRelTol(1e-12);
    cg.SetMaxIter(200);
    cg.SetPrintLevel(1);
    cg.Mult(rhs, x);

    gf.Distribute(x);

    // Restore original values for non-restricted DOFs (belt-and-suspenders)
    if (restricted) {
        for (int i = 0; i < gf.Size(); i++) {
            if (!restricted_dofs.count(i)) {
                gf(i) = gf_orig(i);
            }
        }
    }

    // Restore original scale
    gf *= global_max;
}

// =============================================================================
// Internal: anisotropic solve
// =============================================================================

static void SmoothInternalAniso(ParFiniteElementSpace& fes,
                                ParGridFunction& gf,
                                MatrixCoefficient& sigma2_mat,
                                const std::vector<int>& restrict_attrs) {
    MPI_Comm comm = fes.GetComm();
    bool restricted = !restrict_attrs.empty();

    // Normalize: find global max absolute value
    real_t local_max = 0.0;
    for (int i = 0; i < gf.Size(); i++) {
        local_max = std::max(local_max, std::abs(gf(i)));
    }
    real_t global_max = 0.0;
    MPI_Allreduce(&local_max, &global_max, 1, MFEM_MPI_REAL_T, MPI_MAX, comm);

    if (global_max == 0.0) return;

    gf /= global_max;

    // Save original values for non-restricted DOFs
    Vector gf_orig;
    std::set<int> restricted_dofs;
    if (restricted) {
        gf_orig = gf;
        restricted_dofs = GetRestrictedDofs(fes, restrict_attrs);
    }

    // Assemble mass matrix M (full domain)
    ParBilinearForm mass(&fes);
    mass.AddDomainIntegrator(new MassIntegrator());
    mass.Assemble();
    mass.Finalize();

    // Assemble diffusion matrix K with anisotropic σ² matrix coefficient (full domain)
    ParBilinearForm diff(&fes);
    diff.AddDomainIntegrator(new DiffusionIntegrator(sigma2_mat));
    diff.Assemble();
    diff.Finalize();

    // A = M + K
    std::unique_ptr<HypreParMatrix> M(mass.ParallelAssemble());
    std::unique_ptr<HypreParMatrix> K(diff.ParallelAssemble());
    std::unique_ptr<HypreParMatrix> A(Add(1.0, *M, 1.0, *K));

    // rhs = M * u
    HypreParVector rhs(comm, fes.GlobalTrueVSize(), fes.GetTrueDofOffsets());
    HypreParVector x(comm, fes.GlobalTrueVSize(), fes.GetTrueDofOffsets());
    gf.ParallelProject(x);
    M->Mult(x, rhs);

    // For restricted smoothing, impose Dirichlet-like conditions on
    // non-restricted DOFs
    if (restricted) {
        Array<int> nr_tdofs = GetNonRestrictedTrueDofs(fes, restrict_attrs);
        A->EliminateRowsCols(nr_tdofs, x, rhs);
    }

    // Solve with AMG-preconditioned CG
    x = 0.0;
    HypreBoomerAMG amg(*A);
    amg.SetPrintLevel(0);
    CGSolver cg(comm);
    cg.SetOperator(*A);
    cg.SetPreconditioner(amg);
    cg.SetRelTol(1e-12);
    cg.SetMaxIter(200);
    cg.SetPrintLevel(1);
    cg.Mult(rhs, x);

    gf.Distribute(x);

    // Restore original values for non-restricted DOFs (belt-and-suspenders)
    if (restricted) {
        for (int i = 0; i < gf.Size(); i++) {
            if (!restricted_dofs.count(i)) {
                gf(i) = gf_orig(i);
            }
        }
    }

    gf *= global_max;
}

// =============================================================================
// Public API
// =============================================================================

void HelmholtzSmooth(ParFiniteElementSpace& fes,
                     ParGridFunction& gf,
                     const ParGridFunction& vp_gf,
                     real_t alpha, real_t freq,
                     const std::vector<int>& restrict_attrs) {
    real_t alpha_over_f = alpha / freq;
    VpSquaredCoefficient sigma2_coeff(vp_gf, alpha_over_f);
    SmoothInternal(fes, gf, sigma2_coeff, restrict_attrs);
}

void HelmholtzSmooth(ParFiniteElementSpace& fes,
                     ParGridFunction& gf,
                     real_t sigma,
                     const std::vector<int>& restrict_attrs) {
    real_t sigma2 = sigma * sigma;
    ConstantCoefficient sigma2_coeff(sigma2);
    SmoothInternal(fes, gf, sigma2_coeff, restrict_attrs);
}

void HelmholtzSmoothAniso(ParFiniteElementSpace& fes,
                           ParGridFunction& gf,
                           const ParGridFunction& vp_gf,
                           real_t alpha_x, real_t alpha_y,
                           real_t freq,
                           const std::vector<int>& restrict_attrs) {
    AnisoVpSquaredCoefficient sigma2_mat(vp_gf, alpha_x / freq, alpha_y / freq);
    SmoothInternalAniso(fes, gf, sigma2_mat, restrict_attrs);
}

void HelmholtzSmoothAniso(ParFiniteElementSpace& fes,
                           ParGridFunction& gf,
                           real_t sigma_x, real_t sigma_y,
                           const std::vector<int>& restrict_attrs) {
    AnisoConstantCoefficient sigma2_mat(sigma_x * sigma_x, sigma_y * sigma_y);
    SmoothInternalAniso(fes, gf, sigma2_mat, restrict_attrs);
}

}  // namespace SEM
