/**
 * @file PointFinder.cpp
 * @brief Custom point location implementation (replaces FindPointsGSLIB)
 *
 * This implementation works with any MFEM precision (single or double)
 * and does not depend on GSLIB.
 */

#include "util/PointFinder.hpp"
#include <algorithm>
#include <limits>

namespace SEM {

// =============================================================================
// PointFinder
// =============================================================================

PointFinder::PointFinder(MPI_Comm comm)
    : comm_(comm)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &num_procs_);
}

void PointFinder::Setup(ParMesh& pmesh, real_t newton_tol, int newton_max_iter) {
    pmesh_ = &pmesh;
    dim_ = pmesh.SpaceDimension();
    newton_tol_ = newton_tol;
    newton_max_iter_ = newton_max_iter;

    // Ensure mesh has nodes for high-order geometry
    pmesh_->EnsureNodes();

    // Precompute element bounding boxes for quick rejection
    int ne = pmesh_->GetNE();
    elem_bb_min_.resize(ne);
    elem_bb_max_.resize(ne);

    for (int e = 0; e < ne; e++) {
        elem_bb_min_[e].SetSize(dim_);
        elem_bb_max_[e].SetSize(dim_);
        GetElementBBox(e, elem_bb_min_[e], elem_bb_max_[e]);
    }
    bbox_computed_ = true;
}

void PointFinder::FindPoints(const Vector& points) {
    MFEM_VERIFY(pmesh_ != nullptr, "PointFinder::Setup must be called first");

    int npts = points.Size() / dim_;

    // Allocate result arrays
    elem_.SetSize(npts);
    proc_.SetSize(npts);
    code_.SetSize(npts);
    ref_pos_.SetSize(npts * dim_);

    // Initialize with "not found"
    for (int i = 0; i < npts; i++) {
        elem_[i] = 0;
        proc_[i] = 0;
        code_[i] = 2;  // Not found
    }
    ref_pos_ = 0.0;

    // For each point, search local mesh
    for (int i = 0; i < npts; i++) {
        Vector phys_pt(dim_);
        for (int d = 0; d < dim_; d++) {
            phys_pt[d] = points[i * dim_ + d];
        }

        int local_elem;
        Vector local_ref(dim_);

        if (SearchLocalMesh(phys_pt, local_elem, local_ref)) {
            elem_[i] = local_elem;
            proc_[i] = rank_;
            code_[i] = 0;  // Found inside
            for (int d = 0; d < dim_; d++) {
                ref_pos_[i * dim_ + d] = local_ref[d];
            }
        }
    }

    // Now communicate results across all ranks
    // Each rank broadcasts its findings, and we keep the first valid result
    // (in case a point is on shared boundary)

    if (num_procs_ > 1) {
        // Gather all results to determine ownership
        // Use MPI_Allreduce with custom operation to find the owning rank

        // First, gather all codes
        std::vector<unsigned int> all_codes(npts * num_procs_);
        MPI_Allgather(code_.GetData(), npts, MPI_UNSIGNED,
                      all_codes.data(), npts, MPI_UNSIGNED, comm_);

        // Gather all elem indices
        std::vector<unsigned int> all_elems(npts * num_procs_);
        MPI_Allgather(elem_.GetData(), npts, MPI_UNSIGNED,
                      all_elems.data(), npts, MPI_UNSIGNED, comm_);

        // Gather all reference positions
        std::vector<real_t> all_refs(npts * dim_ * num_procs_);
        MPI_Allgather(ref_pos_.GetData(), npts * dim_, MPITypeMap<real_t>::mpi_type,
                      all_refs.data(), npts * dim_, MPITypeMap<real_t>::mpi_type, comm_);

        // For each point, find the first rank that found it
        for (int i = 0; i < npts; i++) {
            bool found = false;
            for (int r = 0; r < num_procs_; r++) {
                int idx = r * npts + i;
                if (all_codes[idx] != 2) {  // Found by rank r
                    elem_[i] = all_elems[idx];
                    proc_[i] = r;
                    code_[i] = all_codes[idx];
                    for (int d = 0; d < dim_; d++) {
                        ref_pos_[i * dim_ + d] = all_refs[r * npts * dim_ + i * dim_ + d];
                    }
                    found = true;
                    break;
                }
            }
            if (!found) {
                code_[i] = 2;  // Not found on any rank
            }
        }
    }
}

bool PointFinder::SearchLocalMesh(const Vector& phys_pt, int& elem, Vector& ref_pt) {
    int ne = pmesh_->GetNE();

    for (int e = 0; e < ne; e++) {
        // Quick bounding box rejection test
        if (bbox_computed_) {
            bool outside = false;
            for (int d = 0; d < dim_; d++) {
                if (phys_pt[d] < elem_bb_min_[e][d] - newton_tol_ ||
                    phys_pt[d] > elem_bb_max_[e][d] + newton_tol_) {
                    outside = true;
                    break;
                }
            }
            if (outside) continue;
        }

        // Newton iteration to find reference coordinates
        if (NewtonSolve(e, phys_pt, ref_pt)) {
            elem = e;
            return true;
        }
    }

    return false;
}

bool PointFinder::NewtonSolve(int elem, const Vector& phys_pt, Vector& ref_pt) {
    ElementTransformation* T = pmesh_->GetElementTransformation(elem);
    Geometry::Type geom = pmesh_->GetElementGeometry(elem);

    // Initialize reference point at element center
    const IntegrationRule& ir = IntRules.Get(geom, 1);
    IntegrationPoint ip;
    ip.x = ip.y = ip.z = 0.0;

    // Get center of reference element
    for (int i = 0; i < ir.GetNPoints(); i++) {
        const IntegrationPoint& ip_i = ir.IntPoint(i);
        ip.x += ip_i.x;
        ip.y += ip_i.y;
        ip.z += ip_i.z;
    }
    ip.x /= ir.GetNPoints();
    ip.y /= ir.GetNPoints();
    ip.z /= ir.GetNPoints();

    Vector x_ref(dim_);
    x_ref[0] = ip.x;
    if (dim_ >= 2) x_ref[1] = ip.y;
    if (dim_ >= 3) x_ref[2] = ip.z;

    Vector x_phys(dim_);
    Vector residual(dim_);
    DenseMatrix J(dim_);
    DenseMatrix Jinv(dim_);

    // Newton iteration
    for (int iter = 0; iter < newton_max_iter_; iter++) {
        // Set integration point
        ip.x = x_ref[0];
        if (dim_ >= 2) ip.y = x_ref[1];
        if (dim_ >= 3) ip.z = x_ref[2];

        T->SetIntPoint(&ip);

        // Get physical coordinates at current reference point
        T->Transform(ip, x_phys);

        // Compute residual: r = x_target - x(xi)
        subtract(phys_pt, x_phys, residual);

        // Check convergence
        real_t res_norm = residual.Norml2();
        if (res_norm < newton_tol_) {
            ref_pt = x_ref;
            // Check if inside reference element
            return IsInsideReferenceElement(ref_pt, geom);
        }

        // Get Jacobian
        const DenseMatrix& Jac = T->Jacobian();
        J = Jac;

        // Invert Jacobian
        CalcInverse(J, Jinv);

        // Newton update: xi_new = xi + J^{-1} * r
        Vector delta(dim_);
        Jinv.Mult(residual, delta);
        x_ref += delta;

        // Clamp to reasonable bounds to prevent divergence
        for (int d = 0; d < dim_; d++) {
            x_ref[d] = std::max(real_t(-2.0), std::min(real_t(2.0), x_ref[d]));
        }
    }

    // Check if final point is inside even if not fully converged
    ref_pt = x_ref;
    return IsInsideReferenceElement(ref_pt, geom);
}

bool PointFinder::IsInsideReferenceElement(const Vector& ref_pt, Geometry::Type geom) const {
    // Use larger tolerance for single precision
#ifdef MFEM_USE_SINGLE
    const real_t tol = 1e-4;
#else
    const real_t tol = 1e-6;
#endif

    switch (geom) {
        case Geometry::SEGMENT:
            return (ref_pt[0] >= -tol && ref_pt[0] <= 1.0 + tol);

        case Geometry::TRIANGLE:
            return (ref_pt[0] >= -tol && ref_pt[1] >= -tol &&
                    ref_pt[0] + ref_pt[1] <= 1.0 + tol);

        case Geometry::SQUARE:
            return (ref_pt[0] >= -tol && ref_pt[0] <= 1.0 + tol &&
                    ref_pt[1] >= -tol && ref_pt[1] <= 1.0 + tol);

        case Geometry::TETRAHEDRON:
            return (ref_pt[0] >= -tol && ref_pt[1] >= -tol && ref_pt[2] >= -tol &&
                    ref_pt[0] + ref_pt[1] + ref_pt[2] <= 1.0 + tol);

        case Geometry::CUBE:
            return (ref_pt[0] >= -tol && ref_pt[0] <= 1.0 + tol &&
                    ref_pt[1] >= -tol && ref_pt[1] <= 1.0 + tol &&
                    ref_pt[2] >= -tol && ref_pt[2] <= 1.0 + tol);

        case Geometry::PRISM:
            return (ref_pt[0] >= -tol && ref_pt[1] >= -tol &&
                    ref_pt[0] + ref_pt[1] <= 1.0 + tol &&
                    ref_pt[2] >= -tol && ref_pt[2] <= 1.0 + tol);

        case Geometry::PYRAMID:
            // Pyramid has more complex bounds
            {
                real_t z = ref_pt[2];
                real_t bound = 1.0 - z;
                return (ref_pt[0] >= -tol && ref_pt[0] <= bound + tol &&
                        ref_pt[1] >= -tol && ref_pt[1] <= bound + tol &&
                        ref_pt[2] >= -tol && ref_pt[2] <= 1.0 + tol);
            }

        default:
            return false;
    }
}

void PointFinder::GetElementBBox(int elem, Vector& bb_min, Vector& bb_max) {
    Array<int> verts;
    pmesh_->GetElementVertices(elem, verts);

    // Initialize with first vertex
    for (int d = 0; d < dim_; d++) {
        bb_min[d] = std::numeric_limits<real_t>::max();
        bb_max[d] = std::numeric_limits<real_t>::lowest();
    }

    // Get vertex coordinates
    for (int i = 0; i < verts.Size(); i++) {
        const real_t* v = pmesh_->GetVertex(verts[i]);
        for (int d = 0; d < dim_; d++) {
            bb_min[d] = std::min(bb_min[d], v[d]);
            bb_max[d] = std::max(bb_max[d], v[d]);
        }
    }

    // For curved elements, also check interior nodes
    if (pmesh_->GetNodes()) {
        const GridFunction* nodes = pmesh_->GetNodes();
        const FiniteElementSpace* fes = nodes->FESpace();
        Array<int> dofs;
        fes->GetElementDofs(elem, dofs);

        for (int i = 0; i < dofs.Size(); i++) {
            int dof = dofs[i];
            if (dof < 0) dof = -1 - dof;  // Handle negative DOF indices

            for (int d = 0; d < dim_; d++) {
                real_t coord = (*nodes)[fes->DofToVDof(dof, d)];
                bb_min[d] = std::min(bb_min[d], coord);
                bb_max[d] = std::max(bb_max[d], coord);
            }
        }
    }

    // Add small margin for numerical tolerance
    real_t margin = newton_tol_ * 10;
    for (int d = 0; d < dim_; d++) {
        bb_min[d] -= margin;
        bb_max[d] += margin;
    }
}

}  // namespace SEM
