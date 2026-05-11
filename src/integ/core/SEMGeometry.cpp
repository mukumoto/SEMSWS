/**
 * @file SEMGeometry.cpp
 * @brief Implementation of geometry computation for SEM integrators
 */

#include "integ/core/SEMGeometry.hpp"
#include "util/FESOrder.hpp"
#include "general/forall.hpp"

namespace SEM {

// =============================================================================
// SEMGeometry2D Implementation
// =============================================================================

void SEMGeometry2D::Compute(const FiniteElementSpace& fes)
{
    Mesh* mesh = fes.GetMesh();
    ne = mesh->GetNE();
    order = SafeFESOrder(fes);
    ngll = order + 1;

    const int ngll2 = ngll * ngll;
    const int64_t total_pts = (int64_t)ne * ngll2;

    // =========================================================================
    // Allocate storage (with overflow check)
    // =========================================================================
    const int64_t packed_size = 5 * total_pts;  // 5 components: j0, j1, j2, j3, detJ
    const int64_t invJ_size = total_pts * 4;
    MFEM_VERIFY(packed_size <= INT_MAX,
                "SEMGeometry2D::invJ_packed size (" << packed_size << ") exceeds int32 limit. "
                "Reduce elements per GPU or increase GPU count.");
    MFEM_VERIFY(invJ_size <= INT_MAX,
                "SEMGeometry2D::invJ size (" << invJ_size << ") exceeds int32 limit. "
                "Reduce elements per GPU or increase GPU count.");
    MFEM_VERIFY(total_pts <= INT_MAX,
                "SEMGeometry2D::detJ size (" << total_pts << ") exceeds int32 limit. "
                "Reduce elements per GPU or increase GPU count.");
    invJ_packed.SetSize(static_cast<int>(packed_size));  // Packed: [5, ngll, ngll, ne]
    invJ.SetSize(static_cast<int>(invJ_size));  // Legacy AoS: [base*4+0..3] per point
    detJ.SetSize(static_cast<int>(total_pts));

    dxshape.SetSize(ngll * ngll);
    dyshape.SetSize(ngll * ngll);
    dxshape_w.SetSize(ngll * ngll);
    dyshape_w.SetSize(ngll * ngll);

    wx.SetSize(ngll);
    wy.SetSize(ngll);

    // =========================================================================
    // Build GLL integration rules
    // =========================================================================
    IntegrationRules gll_int(0, Quadrature1D::GaussLobatto);
    int exact = 2 * order - 1;
    IntegrationRule irx = gll_int.Get(Geometry::SEGMENT, exact);
    IntegrationRule iry = gll_int.Get(Geometry::SEGMENT, exact);
    IntegrationRule ir2d(irx, iry);

    // =========================================================================
    // Compute 1D quadrature weights
    // Use HostWrite() instead of GetData() to mark host memory as valid
    // This ensures proper host->device sync when Read() is called later
    // =========================================================================
    real_t* wx_data = wx.HostWrite();
    real_t* wy_data = wy.HostWrite();

    for (int i = 0; i < ngll; i++) {
        wx_data[i] = irx.IntPoint(i).weight;
        wy_data[i] = iry.IntPoint(i).weight;
    }

    // =========================================================================
    // Compute lexicographic ordering (H1_QuadrilateralElement is a NodalFiniteElement)
    // =========================================================================
    lex_ordering.SetSize(ngll2);
    H1_QuadrilateralElement quad_fe(order);
    const Array<int>& lex_ref = quad_fe.GetLexicographicOrdering();
    for (int idx = 0; idx < ngll2; idx++) {
        lex_ordering[idx] = lex_ref[idx];
    }

    // =========================================================================
    // Compute shape function derivatives
    // =========================================================================
    int dof = quad_fe.GetDof();
    DenseMatrix dshape_mat(dof, 2);

    real_t* dx = dxshape.HostWrite();
    real_t* dy = dyshape.HostWrite();
    real_t* dxw = dxshape_w.HostWrite();
    real_t* dyw = dyshape_w.HostWrite();
    const int* lex = lex_ordering.GetData();

    // x-direction: Store in ROW-MAJOR order
    for (int i = 0; i < ngll; i++) {
        const IntegrationPoint& ip = ir2d.IntPoint(i);
        quad_fe.CalcDShape(ip, dshape_mat);
        for (int j = 0; j < ngll; j++) {
            dx[i * ngll + j] = dshape_mat(lex[j], 0);
            dxw[i * ngll + j] = dshape_mat(lex[j], 0) * irx.IntPoint(i).weight;
        }
    }

    // y-direction: Store in ROW-MAJOR order
    for (int i = 0; i < ngll; i++) {
        const IntegrationPoint& ip = ir2d.IntPoint(i * ngll);
        quad_fe.CalcDShape(ip, dshape_mat);
        for (int j = 0; j < ngll; j++) {
            dy[i * ngll + j] = dshape_mat(lex[j * ngll], 1);
            dyw[i * ngll + j] = dshape_mat(lex[j * ngll], 1) * iry.IntPoint(i).weight;
        }
    }

    // =========================================================================
    // Compute geometry (Jacobian data) - both packed and legacy layouts
    //
    // Packed layout (new): invJ_packed[5, ngll, ngll, ne]
    //   Components: 0=j00, 1=j01, 2=j10, 3=j11, 4=detJ
    //   Access: invJ_packed(comp, ix, iy, e) = data[comp + 5*(ix + ngll*(iy + ngll*e))]
    //
    // Legacy AoS layout: invJ[base*4 + 0] = Jinv(0,0), [+1] = Jinv(1,1),
    //                    [+2] = Jinv(0,1), [+3] = Jinv(1,0)
    // =========================================================================
    real_t* packed_data = invJ_packed.HostWrite();
    real_t* invJ_data = invJ.HostWrite();
    real_t* detJ_data = detJ.HostWrite();

    int stride_e = ngll * ngll;

    for (int e = 0; e < ne; e++) {
        ElementTransformation* Tr = fes.GetElementTransformation(e);

        for (int iy = 0; iy < ngll; iy++) {
            for (int ix = 0; ix < ngll; ix++) {
                int ip_idx = iy * ngll + ix;
                const IntegrationPoint& ip = ir2d.IntPoint(ip_idx);
                Tr->SetIntPoint(&ip);

                const DenseMatrix& Jinv = Tr->InverseJacobian();
                const real_t w = Tr->Weight();

                // Legacy AoS layout
                int base_idx = e * stride_e + iy * ngll + ix;
                int invJ_base = base_idx * 4;

                invJ_data[invJ_base + 0] = Jinv(0, 0);  // ∂ξ/∂x
                invJ_data[invJ_base + 1] = Jinv(1, 1);  // ∂η/∂y
                invJ_data[invJ_base + 2] = Jinv(0, 1);  // ∂ξ/∂y
                invJ_data[invJ_base + 3] = Jinv(1, 0);  // ∂η/∂x
                detJ_data[base_idx] = w;

                // Packed layout: [5, ngll, ngll, ne] column-major
                // packed(comp, ix, iy, e) = data[comp + 5*(ix + ngll*(iy + ngll*e))]
                int packed_base = 5 * (ix + ngll * (iy + ngll * e));
                packed_data[packed_base + 0] = Jinv(0, 0);  // j00
                packed_data[packed_base + 1] = Jinv(1, 1);  // j01 (swapped for kernel compatibility)
                packed_data[packed_base + 2] = Jinv(0, 1);  // j10 (swapped for kernel compatibility)
                packed_data[packed_base + 3] = Jinv(1, 0);  // j11
                packed_data[packed_base + 4] = w;           // detJ
            }
        }
    }

    // Enable device memory AFTER all data is written
    // This ensures the first Read() call will sync host->device
    EnableDevice();

    // Force host→device sync for GPU builds
    SyncToDevice();
}

size_t SEMGeometry2D::MemoryUsage() const
{
    size_t bytes = 0;
    bytes += invJ_packed.Size() * sizeof(real_t);
    bytes += invJ.Size() * sizeof(real_t);
    bytes += detJ.Size() * sizeof(real_t);
    bytes += (dxshape.Size() + dyshape.Size() + dxshape_w.Size() + dyshape_w.Size()) * sizeof(real_t);
    bytes += (wx.Size() + wy.Size()) * sizeof(real_t);
    bytes += lex_ordering.Size() * sizeof(int);
    return bytes;
}


// =============================================================================
// SEMGeometry3D Implementation
// =============================================================================

void SEMGeometry3D::Compute(const FiniteElementSpace& fes)
{
    Mesh* mesh = fes.GetMesh();
    ne = mesh->GetNE();
    order = SafeFESOrder(fes);
    ngll = order + 1;

    const int ngll3 = ngll * ngll * ngll;
    const int64_t total_pts = (int64_t)ne * ngll3;

    // =========================================================================
    // Allocate storage (packed layout only, with overflow check)
    // =========================================================================
    const int64_t packed_size = 10 * total_pts;
    if (packed_size > INT_MAX)
    {
        mfem::err << "SEMGeometry3D::invJ_packed size (" << packed_size << ") exceeds int32 limit.\n"
                  << "Reduce elements per GPU (ne=" << ne << ") or increase GPU count.\n"
                  << "Maximum elements per GPU/CPU for NGLL:\n"
                  << "  NGLL | MAX NE\n"
                  << "  -----|----------\n"
                  << "    5  | 1,717,986\n"
                  << "    6  |   994,209\n"
                  << "    7  |   626,379\n"
                  << "    8  |   421,875\n";
        MFEM_ABORT("");
    }

    invJ_packed.SetSize(static_cast<int>(packed_size));
    invJ_packed.UseDevice(true);

    // Shape derivatives (same for all directions in tensor product element)
    dxshape.SetSize(ngll * ngll);
    dxshape_w.SetSize(ngll * ngll);

    // Quadrature weights (same for all directions in tensor product element)
    wgll.SetSize(ngll);

    wgllwgll_xy.SetSize(ngll * ngll);
    wgllwgll_xz.SetSize(ngll * ngll);
    wgllwgll_yz.SetSize(ngll * ngll);

    // =========================================================================
    // Build GLL integration rules
    // =========================================================================
    IntegrationRules gll_int(0, Quadrature1D::GaussLobatto);
    int exact = 2 * order - 1;
    IntegrationRule irx = gll_int.Get(Geometry::SEGMENT, exact);
    IntegrationRule iry = gll_int.Get(Geometry::SEGMENT, exact);
    IntegrationRule irz = gll_int.Get(Geometry::SEGMENT, exact);
    IntegrationRule ir3d(irx, iry, irz);

    // =========================================================================
    // Compute 1D quadrature weights (same for all directions)
    // Use HostWrite() instead of GetData() to mark host memory as valid
    // This ensures proper host->device sync when Read() is called later
    // =========================================================================
    real_t* w_data = wgll.HostWrite();

    for (int i = 0; i < ngll; i++) {
        w_data[i] = irx.IntPoint(i).weight;
    }

    // Pre-compute weight products
    real_t* wxy = wgllwgll_xy.HostWrite();
    real_t* wxz = wgllwgll_xz.HostWrite();
    real_t* wyz = wgllwgll_yz.HostWrite();
    for (int iy = 0; iy < ngll; iy++) {
        for (int ix = 0; ix < ngll; ix++) {
            wxy[iy * ngll + ix] = w_data[ix] * w_data[iy];
        }
    }
    for (int iz = 0; iz < ngll; iz++) {
        for (int ix = 0; ix < ngll; ix++) {
            wxz[iz * ngll + ix] = w_data[ix] * w_data[iz];
        }
    }
    for (int iz = 0; iz < ngll; iz++) {
        for (int iy = 0; iy < ngll; iy++) {
            wyz[iz * ngll + iy] = w_data[iy] * w_data[iz];
        }
    }

    // =========================================================================
    // Compute lexicographic ordering (H1_HexahedronElement is a NodalFiniteElement)
    // =========================================================================
    lex_ordering.SetSize(ngll3);
    H1_HexahedronElement hex_fe(order);
    const Array<int>& lex_ref = hex_fe.GetLexicographicOrdering();
    for (int idx = 0; idx < ngll3; idx++) {
        lex_ordering[idx] = lex_ref[idx];
    }

    // =========================================================================
    // Compute shape function derivatives
    // Note: For tensor product elements, all directions use the same 1D shape functions
    // =========================================================================
    int dof = hex_fe.GetDof();
    DenseMatrix dshape_mat(dof, 3);

    real_t* dshape = dxshape.HostWrite();
    real_t* dshape_weighted = dxshape_w.HostWrite();
    const int* lex = lex_ordering.GetData();

    // Compute 1D shape derivatives (same for all directions)
    // We use x-direction at iy=0, iz=0 (i.e., IntegrationPoint index = ix)
    for (int i = 0; i < ngll; i++) {
        const IntegrationPoint& ip = ir3d.IntPoint(i);
        hex_fe.CalcDShape(ip, dshape_mat);
        for (int j = 0; j < ngll; j++) {
            dshape[i * ngll + j] = dshape_mat(lex[j], 0);
            dshape_weighted[i * ngll + j] = dshape_mat(lex[j], 0) * w_data[i];
        }
    }

    // =========================================================================
    // GPU-accelerated geometry computation using MFEM GeometricFactors
    // Compute inverse Jacobian and detJ directly into packed layout
    // =========================================================================

    // Get Jacobian matrix J and detJ from GeometricFactors (GPU-computed)
    const GeometricFactors* geom = mesh->GetGeometricFactors(
        ir3d, GeometricFactors::JACOBIANS | GeometricFactors::DETERMINANTS);

    // Compute inverse Jacobian on GPU from J matrix, write directly to packed layout
    // GeometricFactors::J layout: [NQ x SDIM x DIM x NE] = [ngll^3 x 3 x 3 x ne]
    const int NQ = ngll3;
    const int NE = ne;
    const auto d_J = Reshape(geom->J.Read(), NQ, 3, 3, NE);
    const auto d_detJ = geom->detJ.Read();
    auto d_packed = invJ_packed.Write();

    mfem::forall(NE * NQ, [=] MFEM_HOST_DEVICE (int idx)
    {
        const int e = idx / NQ;
        const int q = idx % NQ;

        // Load Jacobian matrix from GeometricFactors
        // J layout is column-major: J(q, row, col, e)
        const real_t J00 = d_J(q, 0, 0, e), J01 = d_J(q, 0, 1, e), J02 = d_J(q, 0, 2, e);
        const real_t J10 = d_J(q, 1, 0, e), J11 = d_J(q, 1, 1, e), J12 = d_J(q, 1, 2, e);
        const real_t J20 = d_J(q, 2, 0, e), J21 = d_J(q, 2, 1, e), J22 = d_J(q, 2, 2, e);

        // Compute determinant
        const real_t det = J00 * (J11 * J22 - J12 * J21)
                         - J01 * (J10 * J22 - J12 * J20)
                         + J02 * (J10 * J21 - J11 * J20);
        const real_t inv_det = 1.0 / det;

        // Compute inverse Jacobian (adjugate / det) and store directly to packed layout
        // Packed layout: base = 10 * idx, comp 0-8 = invJ, comp 9 = detJ
        const int base = 10 * idx;
        d_packed[base + 0] = (J11 * J22 - J12 * J21) * inv_det;
        d_packed[base + 1] = (J02 * J21 - J01 * J22) * inv_det;
        d_packed[base + 2] = (J01 * J12 - J02 * J11) * inv_det;
        d_packed[base + 3] = (J12 * J20 - J10 * J22) * inv_det;
        d_packed[base + 4] = (J00 * J22 - J02 * J20) * inv_det;
        d_packed[base + 5] = (J02 * J10 - J00 * J12) * inv_det;
        d_packed[base + 6] = (J10 * J21 - J11 * J20) * inv_det;
        d_packed[base + 7] = (J01 * J20 - J00 * J21) * inv_det;
        d_packed[base + 8] = (J00 * J11 - J01 * J10) * inv_det;
        d_packed[base + 9] = d_detJ[idx];
    });

    // Enable device memory for small arrays (computed on CPU)
    wgll.UseDevice(true);
    wgllwgll_xy.UseDevice(true); wgllwgll_xz.UseDevice(true); wgllwgll_yz.UseDevice(true);
    dxshape.UseDevice(true);
    dxshape_w.UseDevice(true);

    // Force host→device sync for small arrays
    wgll.Read();
    wgllwgll_xy.Read(); wgllwgll_xz.Read(); wgllwgll_yz.Read();
    dxshape.Read();
    dxshape_w.Read();
}

size_t SEMGeometry3D::MemoryUsage() const
{
    size_t bytes = 0;
    bytes += invJ_packed.Size() * sizeof(real_t);
    bytes += (dxshape.Size() + dxshape_w.Size()) * sizeof(real_t);
    bytes += wgll.Size() * sizeof(real_t);
    bytes += (wgllwgll_xy.Size() + wgllwgll_xz.Size() + wgllwgll_yz.Size()) * sizeof(real_t);
    bytes += lex_ordering.Size() * sizeof(int);
    return bytes;
}

}  // namespace SEM
