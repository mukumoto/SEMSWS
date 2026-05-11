/**
 * @file ReceiverLocation.cpp
 * @brief ReceiverArray location and interpolation methods
 */

#include "srcrecv/Receiver.hpp"
#include "util/PointFinder.hpp"

namespace SEM {

// =============================================================================
// ReceiverArray - Location and Interpolation
// =============================================================================

void ReceiverArray::LocateReceivers() {
    // Gather all receiver positions
    Vector rec_posi(space_dim_ * num_total_);
    int count = 0;
    for (const auto& entry : receivers_) {
        for (const auto& rec : entry.second) {
            for (int d = 0; d < space_dim_; d++) {
                rec_posi[space_dim_ * count + d] = rec.Position()[d];
            }
            count++;
        }
    }

    // Use custom PointFinder (GSLIB-independent, works with any precision)
    fes_.GetParMesh()->EnsureNodes();
    PointFinder finder(*comm_);
    finder.Setup(*fes_.GetParMesh());
    finder.FindPoints(rec_posi);

    // PointFinder returns results on all ranks after FindPoints
    // - GetElem(): local element index on the owning processor
    // - GetProc(): MPI rank that owns the element
    // - GetCode(): 0=inside, 1=on boundary, 2=not found
    // - GetReferencePosition(): reference coordinates within the element
    const Array<unsigned int>& elem_arr = finder.GetElem();
    const Array<unsigned int>& code_arr = finder.GetCode();
    const Array<unsigned int>& rank_arr = finder.GetProc();
    const Vector& ref_posi = finder.GetReferencePosition();

    // Update receiver data - each receiver is assigned to its owning rank
    count = 0;
    for (auto& entry : receivers_) {
        for (auto& rec : entry.second) {
            if (code_arr[count] == 2) {
                MFEM_ABORT("Receiver not found in mesh: " << rec.Name()
                           << " at position [" << rec.Position()[0]
                           << (space_dim_ > 1 ? ", " + std::to_string(rec.Position()[1]) : "")
                           << (space_dim_ > 2 ? ", " + std::to_string(rec.Position()[2]) : "")
                           << "]");
            }

            Vector ref_pos(space_dim_);
            for (int d = 0; d < space_dim_; d++) {
                ref_pos[d] = ref_posi[count * space_dim_ + d];
            }

            // elem_arr is the LOCAL element index on the owning rank (rank_arr)
            bool is_local = (static_cast<int>(rank_arr[count]) == local_rank_);
            rec.SetLocation(elem_arr[count], rank_arr[count], ref_pos, is_local);

            count++;
        }
    }
}

void ReceiverArray::PrecomputeInterpolation() {
    // Precompute shape functions and DOF indices for each local receiver.
    // This is done ONCE during setup, avoiding repeated CalcShape() calls
    // at every time step (major performance improvement).

    int ne_local = fes_.GetNE();  // Number of local elements

    for (auto& entry : receivers_) {
        for (auto& rec : entry.second) {
            if (!rec.IsLocal()) continue;

            int elem = rec.ElementIndex();

            // Check if element index is valid for this rank
            MFEM_VERIFY(elem >= 0 && elem < ne_local,
                "Invalid element index " << elem << " for receiver " << rec.Name()
                << " on rank " << local_rank_ << " (ne_local=" << ne_local << ")");

            const FiniteElement& fe = *fes_.GetFE(elem);
            int ndof = fe.GetDof();

            // Compute shape functions at receiver reference position
            IntegrationPoint ip;
            if (space_dim_ == 2) {
                ip.Set2(rec.RefPosition()[0], rec.RefPosition()[1]);
            } else {
                ip.Set3(rec.RefPosition()[0], rec.RefPosition()[1], rec.RefPosition()[2]);
            }

            Vector shape(ndof);
            fe.CalcShape(ip, shape);

            // Get element DOF indices
            Array<int> vdofs;
            fes_.GetElementVDofs(elem, vdofs);

            // Cache for reuse at every time step
            rec.CacheInterpolation(shape, vdofs);
        }
    }
}

}  // namespace SEM
