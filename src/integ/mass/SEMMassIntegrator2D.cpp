/**
 * @file SEMMassIntegrator2D.cpp
 * @brief Implementation of unified 2D SEM mass integrator
 *
 * Single kernel with coefficient parameter - no physics-specific duplication.
 */

#include "integ/mass/SEMMassIntegrator2D.hpp"
#include "util/FESOrder.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include <mfem.hpp>

namespace SEM {

// =============================================================================
// SEMMassIntegrator2D Implementation
// =============================================================================

void SEMMassIntegrator2D::AssemblePA(const FiniteElementSpace &fes)
{
    fespace_ = &fes;

    // Get mesh and element info
    Mesh* mesh = fes.GetMesh();
    ne_ = mesh->GetNE();
    order_ = SafeFESOrder(fes);
    ngll_ = order_ + 1;

    // Compute geometry if not already done
    if (!HasGeometry()) {
        ComputeGeometry(fes);
    }

    // Compute gather_map mappings based on field type
    if (is_vector_field_) {
        ComputeVectorGatherMap(fes);
    } else {
        ComputeScalarGatherMap(fes);
    }
}

void SEMMassIntegrator2D::SetVectorField(bool is_vector)
{
    is_vector_field_ = is_vector;

    // Recompute gather_map if already assembled
    if (fespace_ != nullptr) {
        if (is_vector_field_) {
            ComputeVectorGatherMap(*fespace_);
        } else {
            ComputeScalarGatherMap(*fespace_);
        }
    }
}

void SEMMassIntegrator2D::ComputeGeometry(const FiniteElementSpace& fes)
{
    Mesh* mesh = fes.GetMesh();
    const int ngll = ngll_;
    const int ngll2 = ngll * ngll;

    // Build GLL integration rules
    IntegrationRules gll_int(0, Quadrature1D::GaussLobatto);
    int exact = 2 * order_ - 1;
    IntegrationRule irx = gll_int.Get(Geometry::SEGMENT, exact);
    IntegrationRule iry = gll_int.Get(Geometry::SEGMENT, exact);
    IntegrationRule ir2d(irx, iry);

    // =========================================================================
    // GPU-accelerated detJ computation using MFEM GeometricFactors
    // This replaces the CPU serial loop with GPU-parallel computation
    // =========================================================================
    const GeometricFactors* geom = mesh->GetGeometricFactors(
        ir2d, GeometricFactors::DETERMINANTS);

    // Copy detJ from GeometricFactors (already on GPU if CUDA build)
    // GeometricFactors layout: [NQ x NE] where NQ = ngll^2
    detJ_ = geom->detJ;
    detJ_.UseDevice(true);

    // =========================================================================
    // 1D quadrature weights (small arrays, CPU computation is efficient)
    // =========================================================================
    wx_.SetSize(ngll);
    wy_.SetSize(ngll);

    real_t* wx = wx_.HostWrite();
    real_t* wy = wy_.HostWrite();

    for (int i = 0; i < ngll; i++) {
        wx[i] = irx.IntPoint(i).weight;
        wy[i] = iry.IntPoint(i).weight;
    }

    // Enable device memory and sync
    wx_.UseDevice(true);
    wy_.UseDevice(true);
    wx_.Read();
    wy_.Read();

    // =========================================================================
    // Lexicographic ordering (small array, CPU computation is efficient)
    // =========================================================================
    lex_ordering_.SetSize(ngll2);
    H1_QuadrilateralElement quad_fe(order_);
    const Array<int>& lex_ref = quad_fe.GetLexicographicOrdering();
    for (int idx = 0; idx < ngll2; idx++) {
        lex_ordering_[idx] = lex_ref[idx];
    }
}

void SEMMassIntegrator2D::ComputeScalarGatherMap(const FiniteElementSpace& fes)
{
    const int ngll2 = ngll_ * ngll_;

    gather_map_.SetSize(ne_ * ngll2);

    // H1_QuadrilateralElement is a NodalFiniteElement
    H1_QuadrilateralElement quad_fe(order_);
    const Array<int>& lex_ref = quad_fe.GetLexicographicOrdering();

    Array<int> dof_ids;
    for (int e = 0; e < ne_; e++) {
        fes.GetElementDofs(e, dof_ids);
        for (int iy = 0; iy < ngll_; iy++) {
            for (int ix = 0; ix < ngll_; ix++) {
                int local_idx = (e * ngll_ + iy) * ngll_ + ix;
                int lex_local = ix + iy * ngll_;
                int lex_idx = lex_ref[lex_local];
                gather_map_[local_idx] = dof_ids[lex_idx];
            }
        }
    }

    // Enable device memory AFTER all data is written
    gather_map_.GetMemory().UseDevice(true);

    // Force host→device sync for GPU builds
    gather_map_.Read();
}

void SEMMassIntegrator2D::ComputeVectorGatherMap(const FiniteElementSpace& fes)
{
    const int ngll2 = ngll_ * ngll_;

    gather_map_x_.SetSize(ne_ * ngll2);
    gather_map_y_.SetSize(ne_ * ngll2);

    // H1_QuadrilateralElement is a NodalFiniteElement
    H1_QuadrilateralElement quad_fe(order_);
    const Array<int>& lex_ref = quad_fe.GetLexicographicOrdering();

    int dof = quad_fe.GetDof();

    Array<int> dof_ids;
    for (int e = 0; e < ne_; e++) {
        fes.GetElementVDofs(e, dof_ids);

        for (int iy = 0; iy < ngll_; iy++) {
            for (int ix = 0; ix < ngll_; ix++) {
                int local_idx = (e * ngll_ + iy) * ngll_ + ix;
                int lex_local = ix + iy * ngll_;
                int lex_x_idx = lex_ref[lex_local];
                int lex_y_idx = lex_x_idx + dof;  // y-component offset by dof

                gather_map_x_[local_idx] = dof_ids[lex_x_idx];
                gather_map_y_[local_idx] = dof_ids[lex_y_idx];
            }
        }
    }

    // Enable device memory AFTER all data is written
    gather_map_x_.GetMemory().UseDevice(true);
    gather_map_y_.GetMemory().UseDevice(true);

    // Force host→device sync for GPU builds
    gather_map_x_.Read();
    gather_map_y_.Read();
}

void SEMMassIntegrator2D::AssembleDiagonalPA(Vector& diag, const Vector& coef) const
{
    SEM_DISPATCH_NGLL(ngll_, AssembleDiagonalPA_Opt, diag, coef);
}

template<int NGLL>
void SEMMassIntegrator2D::AssembleDiagonalPA_Opt(Vector& diag, const Vector& coef) const
{
    constexpr int NGLL_SQ = NGLL * NGLL;
    const int ne = ne_;

    // Get device pointers
    auto d_c = coef.Read();
    auto d_detJ = detJ_.Read();
    auto d_wx = wx_.Read();
    auto d_wy = wy_.Read();
    auto d_diag = diag.ReadWrite();

    if (is_vector_field_) {
        auto d_gather_map_x = gather_map_x_.Read();
        auto d_gather_map_y = gather_map_y_.Read();

        mfem::forall(ne, [=] MFEM_HOST_DEVICE (int ei)
        {
            const int elem_offset = ei * NGLL_SQ;

            for (int iy = 0; iy < NGLL; iy++) {
                for (int ix = 0; ix < NGLL; ix++) {
                    int gll_idx = iy * NGLL + ix;
                    int local_idx = (ei * NGLL + iy) * NGLL + ix;

                    // diag += coef * detJ * wx * wy
                    real_t val = d_c[elem_offset + gll_idx] *
                                 d_detJ[elem_offset + gll_idx] *
                                 d_wx[ix] * d_wy[iy];

                    AtomicAdd(d_diag[d_gather_map_x[local_idx]], val);
                    AtomicAdd(d_diag[d_gather_map_y[local_idx]], val);
                }
            }
        });
    } else {
        auto d_gather_map = gather_map_.Read();

        mfem::forall(ne, [=] MFEM_HOST_DEVICE (int ei)
        {
            const int elem_offset = ei * NGLL_SQ;

            for (int iy = 0; iy < NGLL; iy++) {
                for (int ix = 0; ix < NGLL; ix++) {
                    int gll_idx = iy * NGLL + ix;
                    int local_idx = (ei * NGLL + iy) * NGLL + ix;

                    real_t val = d_c[elem_offset + gll_idx] *
                                 d_detJ[elem_offset + gll_idx] *
                                 d_wx[ix] * d_wy[iy];

                    AtomicAdd(d_diag[d_gather_map[local_idx]], val);
                }
            }
        });
    }
}

size_t SEMMassIntegrator2D::MemoryUsage() const
{
    size_t bytes = 0;
    bytes += detJ_.Size() * sizeof(real_t);
    bytes += (wx_.Size() + wy_.Size()) * sizeof(real_t);
    bytes += lex_ordering_.Size() * sizeof(int);
    bytes += gather_map_.Size() * sizeof(int);
    bytes += gather_map_x_.Size() * sizeof(int);
    bytes += gather_map_y_.Size() * sizeof(int);
    return bytes;
}

// Explicit template instantiations
#define INSTANTIATE_MASS_2D(NGLL) \
    template void SEMMassIntegrator2D::AssembleDiagonalPA_Opt<NGLL>(Vector&, const Vector&) const;

SEM_INSTANTIATE_NGLL_2D(INSTANTIATE_MASS_2D)

#undef INSTANTIATE_MASS_2D

}  // namespace SEM
