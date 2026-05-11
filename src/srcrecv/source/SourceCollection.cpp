/**
 * @file SourceCollection.cpp
 * @brief PointSourceCollection: batch PointFinder setup + assembly
 *
 * Workflow:
 *   1. Add*() — register sources (stores metadata, no PointFinder)
 *   2. Setup() — one batch PointFinder, compute shape/DOF for local only
 *   3. Assemble() — inject sources into RHS
 */

#include "srcrecv/Source.hpp"
#include "util/PointFinder.hpp"
#include <iostream>

namespace SEM {

// =============================================================================
// PointSourceCollection
// =============================================================================

PointSourceCollection::PointSourceCollection(ParFiniteElementSpace* fes, MPI_Comm comm)
    : fes_(fes), comm_(comm)
{
}

// =============================================================================
// Source registration (Phase 1: no PointFinder)
// =============================================================================

void PointSourceCollection::AddSingleForce(int id, const Vector& position)
{
    sources_.push_back(std::make_unique<SingleForceSource>(
        id, fes_, position));
    setup_done_ = false;
}

void PointSourceCollection::AddSingleForceAcoustic(int id, const Vector& position,
                                                    const MaterialField& kappa)
{
    sources_.push_back(std::make_unique<SingleForceSource>(
        id, fes_, position, kappa));
    setup_done_ = false;
}

void PointSourceCollection::AddSingleForceAcoustic3D(int id, const Vector& position,
                                                      const MaterialField3D& kappa)
{
    sources_.push_back(std::make_unique<SingleForceSource>(
        id, fes_, position, kappa));
    setup_done_ = false;
}

void PointSourceCollection::AddMomentTensor(int id, const Vector& position,
                                             const DenseMatrix& moment)
{
    sources_.push_back(std::make_unique<MomentTensorSource>(
        id, fes_, position, moment));
    setup_done_ = false;
}

// =============================================================================
// Setup (Phase 2: batch PointFinder + local-only initialization)
// =============================================================================

void PointSourceCollection::Setup() {
    if (sources_.empty()) {
        setup_done_ = true;
        return;
    }

    int dim = fes_->GetParMesh()->SpaceDimension();
    int nsrc = static_cast<int>(sources_.size());

    // Gather all source positions into one Vector [dim * nsrc]
    Vector all_positions(dim * nsrc);
    for (int i = 0; i < nsrc; i++) {
        const Vector& pos = sources_[i]->Position();
        for (int d = 0; d < dim; d++) {
            all_positions[dim * i + d] = pos[d];
        }
    }

    // One batch PointFinder call (collective)
    fes_->GetParMesh()->EnsureNodes();
    PointFinder finder(comm_);
    finder.Setup(*fes_->GetParMesh());
    finder.FindPoints(all_positions);

    const Array<unsigned int>& elem_arr = finder.GetElem();
    const Array<unsigned int>& code_arr = finder.GetCode();
    const Array<unsigned int>& rank_arr = finder.GetProc();
    const Vector& ref_pos = finder.GetReferencePosition();

    int local_rank;
    MPI_Comm_rank(comm_, &local_rank);

    // Check if any source is on element boundary (ref coords near 0 or 1)
#ifdef MFEM_USE_SINGLE
    const real_t boundary_tol = 1e-4;
#else
    const real_t boundary_tol = 1e-6;
#endif

    bool any_on_boundary = false;
    for (int i = 0; i < nsrc; i++) {
        if (code_arr[i] == 2) continue;
        for (int d = 0; d < dim; d++) {
            real_t r = ref_pos[i * dim + d];
            if (r < boundary_tol || r > 1.0 - boundary_tol) {
                any_on_boundary = true;
                break;
            }
        }
        if (any_on_boundary) break;
    }

    if (any_on_boundary) {
        // Compute element size in physical space from the source's element
        int src_owner = static_cast<int>(rank_arr[0]);
        real_t h_min = 0.0;  // minimum element edge length
        if (src_owner == local_rank) {
            ElementTransformation* trans = fes_->GetElementTransformation(elem_arr[0]);
            // Compute element edge lengths along each axis
            h_min = std::numeric_limits<real_t>::max();
            for (int d = 0; d < dim; d++) {
                IntegrationPoint ip_lo, ip_hi;
                ip_lo.x = ip_lo.y = ip_lo.z = 0.5;
                ip_hi.x = ip_hi.y = ip_hi.z = 0.5;
                // Move along axis d: 0 -> 1
                if (d == 0) { ip_lo.x = 0.0; ip_hi.x = 1.0; }
                else if (d == 1) { ip_lo.y = 0.0; ip_hi.y = 1.0; }
                else { ip_lo.z = 0.0; ip_hi.z = 1.0; }
                Vector phys_lo(dim), phys_hi(dim);
                trans->SetIntPoint(&ip_lo);
                trans->Transform(ip_lo, phys_lo);
                trans->SetIntPoint(&ip_hi);
                trans->Transform(ip_hi, phys_hi);
                phys_hi -= phys_lo;
                h_min = std::min(h_min, phys_hi.Norml2());
            }
        }
        MPI_Bcast(&h_min, 1, MPITypeMap<real_t>::mpi_type, src_owner, comm_);

        // Nudge = 0.1% of minimum element edge length
        real_t eps_phys = h_min * 0.001;

        // Per-dimension inward nudge: push each axis toward the interior of
        // the found reference element. MAX-side boundary (r ≈ 1) is nudged
        // by -eps; MIN-side (r ≈ 0) by +eps. Interior coords are unchanged.
        // This keeps sources inside the found element regardless of whether
        // the boundary is an internal element face or the domain outer face.
        Vector nudged_positions(dim * nsrc);
        for (int i = 0; i < nsrc; i++) {
            for (int d = 0; d < dim; d++) {
                real_t r = ref_pos[i * dim + d];
                real_t sign = 0.0;
                if      (r > 1.0 - boundary_tol) sign = -1.0;
                else if (r <       boundary_tol) sign = +1.0;
                nudged_positions[dim * i + d] =
                    all_positions[dim * i + d] + sign * eps_phys;
            }
        }

        if (Mpi::Root()) {
            for (int i = 0; i < nsrc; i++) {
                bool src_on_bnd = false;
                for (int d = 0; d < dim; d++) {
                    real_t r = ref_pos[i * dim + d];
                    if (r < boundary_tol || r > 1.0 - boundary_tol) {
                        src_on_bnd = true; break;
                    }
                }
                if (src_on_bnd) {
                    std::cout << "WARNING: Source " << sources_[i]->GetId()
                              << " is on element boundary. Nudging inward "
                                 "by up to " << eps_phys << " m per axis."
                              << std::endl;
                }
            }
        }

        // Re-run PointFinder with nudged positions
        finder.FindPoints(nudged_positions);
    }

    // Use (possibly updated) finder results
    const Array<unsigned int>& final_elem = finder.GetElem();
    const Array<unsigned int>& final_code = finder.GetCode();
    const Array<unsigned int>& final_rank = finder.GetProc();
    const Vector& final_ref = finder.GetReferencePosition();

    // Set location for each source
    for (int i = 0; i < nsrc; i++) {
        if (final_code[i] == 2) {
            const Vector& spos = sources_[i]->Position();
            MFEM_ABORT("Source not found in mesh: id=" << sources_[i]->GetId()
                       << " at position [" << spos[0]
                       << (dim > 1 ? ", " + std::to_string(spos[1]) : "")
                       << (dim > 2 ? ", " + std::to_string(spos[2]) : "")
                       << "]");
        }

        Vector ref_pos_i(dim);
        for (int d = 0; d < dim; d++) {
            ref_pos_i[d] = final_ref[i * dim + d];
        }

        bool is_local = (static_cast<int>(final_rank[i]) == local_rank);
        sources_[i]->SetLocation(final_elem[i], final_rank[i], ref_pos_i, is_local);

        // Compute shape functions and type-specific initialization for local sources
        if (is_local) {
            sources_[i]->ComputeShapeFunctions();
            sources_[i]->ApplyAcousticScaling();

            // MomentTensorSource needs equivalent force computation
            auto* mt = dynamic_cast<MomentTensorSource*>(sources_[i].get());
            if (mt) {
                mt->ComputeEquivalentForces();
            }
        }
    }

    setup_done_ = true;
}

// =============================================================================
// Runtime methods
// =============================================================================

void PointSourceCollection::Assemble(int step, ParGridFunction& rhs) {
    MFEM_VERIFY(setup_done_, "PointSourceCollection::Setup() must be called before Assemble()");
    for (size_t i = 0; i < sources_.size(); i++) {
        if (active_source_id_ >= 0 && static_cast<int>(i) != active_source_id_) continue;
        sources_[i]->Assemble(step, rhs);
    }
}

void PointSourceCollection::Reset() {
    sources_.clear();
    setup_done_ = false;
}

void PointSourceCollection::SetActiveSource(int id) {
    MFEM_VERIFY(id >= 0 && id < NumSources(),
        "SetActiveSource: id=" << id << " out of range [0, " << NumSources() << ")");
    active_source_id_ = id;
}

void PointSourceCollection::ClearActiveSource() {
    active_source_id_ = -1;
}

}  // namespace SEM
