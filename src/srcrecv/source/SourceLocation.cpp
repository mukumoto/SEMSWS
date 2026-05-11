/**
 * @file SourceLocation.cpp
 * @brief Source location, shape function computation, and assembly
 *
 * PointSourceBase, SingleForceSource, MomentTensorSource implementations.
 * Source location is done externally by PointSourceCollection::Setup()
 * via batch PointFinder (one collective call for all sources).
 */

#include "srcrecv/Source.hpp"
#include "material/Material.hpp"
#include "general/forall.hpp"
#include <cmath>
#include <iostream>

namespace SEM {

// =============================================================================
// Helper: Compute lexicographic ordering for a single element
// =============================================================================

static Array<int> ComputeLexOrdering(int order, int dim)
{
    const int ngll = order + 1;

    if (dim == 2) {
        H1_QuadrilateralElement quad_fe(order);
        const NodalFiniteElement* nodal_fe = &quad_fe;
        const Array<int>& lex_ref = nodal_fe->GetLexicographicOrdering();

        const int total = ngll * ngll;
        Array<int> lex_ordering(total);

        if (lex_ref.Size() > 0) {
            for (int i = 0; i < total; i++) {
                lex_ordering[i] = lex_ref[i];
            }
        } else {
            for (int i = 0; i < total; i++) {
                lex_ordering[i] = i;
            }
        }
        return lex_ordering;
    } else {
        H1_HexahedronElement hex_fe(order);
        const NodalFiniteElement* nodal_fe = &hex_fe;
        const Array<int>& lex_ref = nodal_fe->GetLexicographicOrdering();

        const int total = ngll * ngll * ngll;
        Array<int> lex_ordering(total);

        if (lex_ref.Size() > 0) {
            for (int i = 0; i < total; i++) {
                lex_ordering[i] = lex_ref[i];
            }
        } else {
            for (int i = 0; i < total; i++) {
                lex_ordering[i] = i;
            }
        }
        return lex_ordering;
    }
}

// =============================================================================
// PointSourceBase
// =============================================================================

PointSourceBase::PointSourceBase(int id, ParFiniteElementSpace* fes,
                                 const Vector& position)
    : id_(id), fes_(fes), position_(position)
{
    dim_ = fes_->GetParMesh()->SpaceDimension();
}

void PointSourceBase::SetLocation(int elem, int owner_rank,
                                   const Vector& ref_pos, bool is_local) {
    elem_ = elem;
    owner_rank_ = owner_rank;
    ref_position_ = ref_pos;
    is_local_ = is_local;
    is_found_ = true;
}

void PointSourceBase::ComputeShapeFunctions() {
    if (!is_local_) return;

    const FiniteElement& fe = *fes_->GetFE(elem_);
    ndof_ = fe.GetDof();

    if (Mpi::Root()) {
        std::cout << "Source " << id_
                  << ": owner_rank=" << owner_rank_
                  << ", elem=" << elem_
                  << ", ref=(";
        for (int d = 0; d < dim_; d++) {
            if (d > 0) std::cout << ", ";
            std::cout << ref_position_[d];
        }
        std::cout << ")" << std::endl;
    }

    IntegrationPoint ip;
    if (dim_ == 2) {
        ip.Set2(ref_position_[0], ref_position_[1]);
    } else {
        ip.Set3(ref_position_[0], ref_position_[1], ref_position_[2]);
    }

    shape_.SetSize(ndof_);
    fe.CalcShape(ip, shape_);

    fes_->GetElementVDofs(elem_, source_dofs_);

    // Enable device memory for GPU-compatible source injection
    shape_.UseDevice(true);
    source_dofs_.GetMemory().UseDevice(true);

    // Force host→device sync for GPU builds
    shape_.Read();
    source_dofs_.Read();
}


// =============================================================================
// SingleForceSource
// =============================================================================

SingleForceSource::SingleForceSource(int id, ParFiniteElementSpace* fes,
                                     const Vector& position)
    : PointSourceBase(id, fes, position), is_acoustic_(false)
{
}

SingleForceSource::SingleForceSource(int id, ParFiniteElementSpace* fes,
                                     const Vector& position,
                                     const MaterialField& kappa)
    : PointSourceBase(id, fes, position),
      is_acoustic_(true), kappa_2d_(&kappa)
{
}

SingleForceSource::SingleForceSource(int id, ParFiniteElementSpace* fes,
                                     const Vector& position,
                                     const MaterialField3D& kappa)
    : PointSourceBase(id, fes, position),
      is_acoustic_(true), kappa_3d_(&kappa)
{
}

void SingleForceSource::ApplyAcousticScaling() {
    if (!is_acoustic_ || !is_local_ || !is_found_) return;

    const FiniteElement& fe = *fes_->GetFE(elem_);
    const int order = fe.GetOrder();
    const int ngll = order + 1;

    real_t* shape_data = shape_.HostReadWrite();

    if (kappa_2d_) {
        Array<int> lex_ordering = ComputeLexOrdering(order, 2);
        auto kappa_view = kappa_2d_->ViewHost();

        for (int iy = 0; iy < ngll; iy++) {
            for (int ix = 0; ix < ngll; ix++) {
                int lex_idx = ix + iy * ngll;
                int dof_idx = lex_ordering[lex_idx];
                real_t kappa_val = kappa_view(ix, iy, elem_);
                if (kappa_val > 0.0) {
                    shape_data[dof_idx] /= kappa_val;
                }
            }
        }
    } else if (kappa_3d_) {
        Array<int> lex_ordering = ComputeLexOrdering(order, 3);
        auto kappa_view = kappa_3d_->ViewHost();

        for (int iz = 0; iz < ngll; iz++) {
            for (int iy = 0; iy < ngll; iy++) {
                for (int ix = 0; ix < ngll; ix++) {
                    int lex_idx = ix + (iy + iz * ngll) * ngll;
                    int dof_idx = lex_ordering[lex_idx];
                    real_t kappa_val = kappa_view(ix, iy, iz, elem_);
                    if (kappa_val > 0.0) {
                        shape_data[dof_idx] /= kappa_val;
                    }
                }
            }
        }
    }

    // Force host→device sync after modification
    shape_.Read();
    pressure_scale_ = 1.0;

    // Clear kappa pointers (no longer needed after scaling)
    kappa_2d_ = nullptr;
    kappa_3d_ = nullptr;
}

void SingleForceSource::Assemble(int step, Vector& rhs) {
    if (!is_local_ || !is_found_) return;

    Vector stf_value;
    stf_.GetValue(step, stf_value);

    // Check if source is zero (after STF ends)
    bool all_zero = true;
    for (int c = 0; c < stf_value.Size(); c++) {
        if (stf_value[c] != 0.0) { all_zero = false; break; }
    }
    if (all_zero) return;

    const int ncomp = stf_value.Size();
    const int vdim = fes_->GetVDim();
    const int ndof = ndof_;

    // Get device pointers
    auto rhs_data = rhs.ReadWrite();
    const auto shape_data = shape_.Read();
    const auto dofs_data = source_dofs_.Read();

    if (is_acoustic_) {
        // Scalar field - add directly
        const real_t scale = stf_value[0] * pressure_scale_;
        mfem::forall(ndof, [=] MFEM_HOST_DEVICE (int i) {
            const int dof = dofs_data[i];
            AtomicAdd(rhs_data[dof], shape_data[i] * scale);
        });
    } else {
        // Vector field - process each component
        const int ncomp_actual = std::min(ncomp, vdim);
        for (int c = 0; c < ncomp_actual; c++) {
            const real_t stf_c = stf_value[c];
            if (stf_c == 0.0) continue;
            const int offset = c * ndof;
            mfem::forall(ndof, [=] MFEM_HOST_DEVICE (int i) {
                const int dof = dofs_data[offset + i];
                AtomicAdd(rhs_data[dof], shape_data[i] * stf_c);
            });
        }
    }
}


// =============================================================================
// MomentTensorSource
// =============================================================================

MomentTensorSource::MomentTensorSource(int id, ParFiniteElementSpace* fes,
                                       const Vector& position,
                                       const DenseMatrix& moment)
    : PointSourceBase(id, fes, position), moment_(moment)
{
}

void MomentTensorSource::ComputeEquivalentForces() {
    if (!is_local_ || !is_found_) return;

    const FiniteElement& fe = *fes_->GetFE(elem_);
    ElementTransformation* trans = fes_->GetElementTransformation(elem_);

    IntegrationPoint ip;
    if (dim_ == 2) {
        ip.Set2(ref_position_[0], ref_position_[1]);
    } else {
        ip.Set3(ref_position_[0], ref_position_[1], ref_position_[2]);
    }

    trans->SetIntPoint(&ip);

    DenseMatrix dshape_phys(ndof_, dim_);
    fe.CalcPhysDShape(*trans, dshape_phys);

    equivalent_force_.SetSize(ndof_ * dim_);
    equivalent_force_ = 0.0;

    for (int i = 0; i < ndof_; i++) {
        for (int d = 0; d < dim_; d++) {
            real_t force = 0.0;
            for (int j = 0; j < dim_; j++) {
                force += moment_(d, j) * dshape_phys(i, j);
            }
            equivalent_force_[d * ndof_ + i] = force;
        }
    }

    // Enable device memory for GPU-compatible source injection
    equivalent_force_.UseDevice(true);
    equivalent_force_.Read();
}

void MomentTensorSource::Assemble(int step, Vector& rhs) {
    if (!is_local_ || !is_found_) return;

    const real_t stf_value = stf_.GetValue(step, 0);

    if (stf_value == 0.0) return;

    const int ndof = ndof_;
    const int dim = dim_;

    auto rhs_data = rhs.ReadWrite();
    const auto dofs_data = source_dofs_.Read();
    const auto force_data = equivalent_force_.Read();

    const int total_dofs = ndof * dim;
    mfem::forall(total_dofs, [=] MFEM_HOST_DEVICE (int idx) {
        const int dof = dofs_data[idx];
        AtomicAdd(rhs_data[dof], force_data[idx] * stf_value);
    });
}

}  // namespace SEM
