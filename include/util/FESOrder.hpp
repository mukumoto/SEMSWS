/**
 * @file FESOrder.hpp
 * @brief SafeFESOrder: get the polynomial order of a uniform H1 FE space
 *        without tripping MFEM's per-element assertion on ranks that
 *        happen to have zero local elements.
 *
 * Needed for fluid-solid coupled runs: splitting the parent ParMesh
 * into a fluid and a solid ParSubMesh — each of
 * which inherits the parent's rank distribution. With an unlucky
 * METIS partition a rank can end up with all fluid and zero solid
 * elements (or vice versa). MFEM's ParFiniteElementSpace is perfectly
 * happy in that state — a zero-element FES still participates in
 * MPI collectives through the shared FE collection — but calling
 * `fes.GetOrder(0)` trips `MFEM_VERIFY(i < GetNE(), "Invalid element
 * index")`. SEMSWS integrators used to call `GetOrder(0)` directly
 * and abort on such ranks. `SafeFESOrder` replaces those calls: it
 * reads the order from the first local element when available and
 * falls back to the H1 collection's declared order when the FES is
 * empty on this rank.
 */

#ifndef SEM_UTIL_FES_ORDER_HPP
#define SEM_UTIL_FES_ORDER_HPP

#include <mfem.hpp>

namespace SEM {

/// Return the H1 polynomial order of a uniform FE space. Safe to call
/// on ranks with zero local elements (reads the order from the FE
/// collection instead of asking the first element).
inline int SafeFESOrder(const mfem::FiniteElementSpace& fes)
{
    if (fes.GetNE() > 0) {
        return fes.GetOrder(0);
    }
    const auto* h1 = dynamic_cast<const mfem::H1_FECollection*>(fes.FEColl());
    MFEM_VERIFY(h1,
                "SafeFESOrder called on a FES with zero local elements "
                "whose FE collection is not H1 — cannot recover the "
                "polynomial order without at least one local element.");
    return h1->GetOrder();
}

}  // namespace SEM

#endif  // SEM_UTIL_FES_ORDER_HPP
