/**
 * @file SEMGeometry.hpp
 * @brief Dimension-templated geometry data structures for SEM integrators
 *
 * Provides SEMGeometry<DIM> struct that stores all geometric data needed
 * for SEM integration: inverse Jacobian, determinant, shape derivatives,
 * quadrature weights, and lexicographic ordering.
 */

#ifndef SEM_GEOMETRY_HPP
#define SEM_GEOMETRY_HPP

#include <mfem.hpp>
#include "general/forall.hpp"  // MFEM 4.8 mfem.hpp drops this; needed for mfem::Reshape

namespace SEM {


using namespace mfem;

/**
 * @brief 2D geometry data for SEM integrators
 *
 * Stores inverse Jacobian and detJ in packed layout for memory bandwidth optimization:
 *   invJ_packed[5, ngll, ngll, ne] - interleaved for cache efficiency
 *   Components: 0=j00, 1=j01, 2=j10, 3=j11, 4=detJ
 *
 * Legacy AoS layout (deprecated, kept for compatibility):
 *   invJ[base*4 + 0] = ∂ξ/∂x
 *   invJ[base*4 + 1] = ∂η/∂y
 *   invJ[base*4 + 2] = ∂ξ/∂y
 *   invJ[base*4 + 3] = ∂η/∂x
 */
struct SEMGeometry2D {
    // Packed inverse Jacobian + detJ: [5, ngll, ngll, ne]
    // Layout: invJ_packed(comp, ix, iy, e) where comp: 0-3=invJ, 4=detJ
    // Components: 0=j00, 1=j01, 2=j10, 3=j11, 4=detJ
    Vector invJ_packed;

    // Legacy: Inverse Jacobian: [ne * ngll^2 * 4] in AoS layout (deprecated)
    Vector invJ;

    // Legacy: Determinant of Jacobian: [ne * ngll^2] (deprecated)
    Vector detJ;

    // Shape function derivatives: [ngll * ngll]
    // Note: x/y directions use the same shape functions (tensor product element)
    Vector dxshape, dyshape;         // Unweighted
    Vector dxshape_w, dyshape_w;     // Weighted by quadrature

    // Quadrature weights: [ngll]
    // Note: x/y directions use the same GLL weights (tensor product element)
    Vector wx, wy;

    // Lexicographic ordering: [ngll^2]
    Array<int> lex_ordering;

    // State
    int order = 0;
    int ngll = 0;
    int ne = 0;

    // Cached device pointers (set during SyncToDevice, use d_ prefix)
    const real_t* d_invJ_packed = nullptr;
    const real_t* d_invJ = nullptr;
    const real_t* d_detJ = nullptr;
    const real_t* d_dxshape = nullptr;
    const real_t* d_dyshape = nullptr;
    const real_t* d_dxshape_w = nullptr;
    const real_t* d_dyshape_w = nullptr;
    const real_t* d_wx = nullptr;
    const real_t* d_wy = nullptr;

    /// Check if geometry has been computed
    bool IsValid() const { return invJ_packed.Size() > 0; }

    /// Enable GPU memory for all vectors
    void EnableDevice() {
        invJ_packed.UseDevice(true);
        invJ.UseDevice(true);
        detJ.UseDevice(true);
        dxshape.UseDevice(true);
        dyshape.UseDevice(true);
        dxshape_w.UseDevice(true);
        dyshape_w.UseDevice(true);
        wx.UseDevice(true);
        wy.UseDevice(true);
    }

    /// Force host→device sync and cache device pointers (call after EnableDevice)
    void SyncToDevice() {
        d_invJ_packed = invJ_packed.Read();
        d_invJ = invJ.Read();
        d_detJ = detJ.Read();
        d_dxshape = dxshape.Read();
        d_dyshape = dyshape.Read();
        d_dxshape_w = dxshape_w.Read();
        d_dyshape_w = dyshape_w.Read();
        d_wx = wx.Read();
        d_wy = wy.Read();
    }

    /// Check if device pointers are cached
    bool IsDeviceReady() const { return d_invJ_packed != nullptr; }

    /// Compute geometry from finite element space
    void Compute(const FiniteElementSpace& fes);

    /// Memory usage in bytes
    size_t MemoryUsage() const;

    // =========================================================================
    // View Factory Methods (Zero-Overhead MFEM-style access)
    // =========================================================================

    /// Shape derivative view [ngll, ngll]
    /// Data stored row-major: data[i * ngll + k]
    /// MFEM Reshape column-major: view(a, b) = data[a + b * ngll]
    /// Phase 1 gradient: need data[ix * ngll + k] → use dshape(k, ix)
    /// Phase 2 scatter:  need data[k * ngll + ix] → use dshape_w(ix, k)
    /// Note: Same shape derivatives for x/y directions (tensor product element)
    auto ViewDShape() const { return Reshape(dxshape.Read(), ngll, ngll); }
    auto ViewDShapeW() const { return Reshape(dxshape_w.Read(), ngll, ngll); }

    /// Legacy: Directional shape derivative views (deprecated, use ViewDShape)
    auto ViewDxShape() const { return Reshape(dxshape.Read(), ngll, ngll); }
    auto ViewDyShape() const { return Reshape(dyshape.Read(), ngll, ngll); }
    auto ViewDxShapeW() const { return Reshape(dxshape_w.Read(), ngll, ngll); }
    auto ViewDyShapeW() const { return Reshape(dyshape_w.Read(), ngll, ngll); }

    /// Quadrature weight view [ngll]
    /// Note: Same GLL weights for x/y directions (tensor product element)
    auto ViewWgll() const { return Reshape(wx.Read(), ngll); }

    /// Legacy: Directional quadrature weight views (deprecated, use ViewWgll)
    auto ViewWx() const { return Reshape(wx.Read(), ngll); }
    auto ViewWy() const { return Reshape(wy.Read(), ngll); }

    /// Packed inverse Jacobian + detJ view [5, ngll, ngll, ne]
    /// Access: invJ_packed(comp, ix, iy, e)
    /// Components: 0=j00, 1=j01, 2=j10, 3=j11, 4=detJ
    auto ViewInvJPacked() const { return Reshape(invJ_packed.Read(), 5, ngll, ngll, ne); }

    /// Legacy: DetJ view [ngll^2, ne] - access: detJ(gll_idx, e) (deprecated)
    auto ViewDetJ() const { return Reshape(detJ.Read(), ngll * ngll, ne); }

    /// Legacy: InvJ view [ngll^2 * 4, ne] - AoS layout (deprecated)
    auto ViewInvJ() const { return Reshape(invJ.Read(), ngll * ngll * 4, ne); }
};


/**
 * @brief 3D geometry data for SEM integrators
 *
 * Stores inverse Jacobian and detJ in packed layout for memory bandwidth optimization:
 *   invJ_packed[10, ngll, ngll, ngll, ne] - interleaved for cache efficiency
 *   Components: 0-8 = invJ (row-major), 9 = detJ
 */
struct SEMGeometry3D {
    // Packed inverse Jacobian + detJ: [10, ngll, ngll, ngll, ne]
    // Layout: invJ_packed(comp, ix, iy, iz, e) where comp: 0-8=invJ, 9=detJ
    // Components: 0=j00, 1=j01, 2=j02, 3=j10, 4=j11, 5=j12, 6=j20, 7=j21, 8=j22, 9=detJ
    Vector invJ_packed;

    // Shape function derivatives: [ngll * ngll]
    // Note: x/y/z directions use the same shape functions (tensor product element)
    Vector dxshape;         // Unweighted (same for all directions)
    Vector dxshape_w;       // Weighted by quadrature (same for all directions)

    // Quadrature weights: [ngll]
    // Note: x/y/z directions use the same GLL weights (tensor product element)
    Vector wgll;

    // Pre-computed weight products: [ngll * ngll]
    Vector wgllwgll_xy, wgllwgll_xz, wgllwgll_yz;

    // Lexicographic ordering: [ngll^3]
    Array<int> lex_ordering;

    // State
    int order = 0;
    int ngll = 0;
    int ne = 0;

    // Cached device pointers (set during SyncToDevice, use d_ prefix)
    const real_t* d_invJ_packed = nullptr;
    const real_t* d_dxshape = nullptr;
    const real_t* d_dxshape_w = nullptr;
    const real_t* d_wgll = nullptr;
    const real_t* d_wgllwgll_xy = nullptr;
    const real_t* d_wgllwgll_xz = nullptr;
    const real_t* d_wgllwgll_yz = nullptr;

    /// Check if geometry has been computed
    bool IsValid() const { return invJ_packed.Size() > 0; }

    /// Enable GPU memory for all vectors
    void EnableDevice() {
        invJ_packed.UseDevice(true);
        dxshape.UseDevice(true);
        dxshape_w.UseDevice(true);
        wgll.UseDevice(true);
        wgllwgll_xy.UseDevice(true); wgllwgll_xz.UseDevice(true); wgllwgll_yz.UseDevice(true);
    }

    /// Force host→device sync and cache device pointers (call after EnableDevice)
    void SyncToDevice() {
        // Sync and cache pointers
        d_invJ_packed = invJ_packed.Read();
        d_dxshape = dxshape.Read();
        d_dxshape_w = dxshape_w.Read();
        d_wgll = wgll.Read();
        d_wgllwgll_xy = wgllwgll_xy.Read();
        d_wgllwgll_xz = wgllwgll_xz.Read();
        d_wgllwgll_yz = wgllwgll_yz.Read();
    }

    /// Check if device pointers are cached
    bool IsDeviceReady() const { return d_invJ_packed != nullptr; }

    /// Compute geometry from finite element space
    void Compute(const FiniteElementSpace& fes);

    /// Memory usage in bytes
    size_t MemoryUsage() const;

    // =========================================================================
    // View Factory Methods (Zero-Overhead MFEM-style access)
    // =========================================================================

    /// Shape derivative view [ngll, ngll]
    /// Data stored row-major: data[i * ngll + k]
    /// MFEM Reshape column-major: view(a, b) = data[a + b * ngll]
    /// Phase 1 gradient: need data[ix * ngll + k] → use dshape(k, ix)
    /// Phase 2 scatter:  need data[k * ngll + ix] → use dshape_w(ix, k)
    /// Note: Same shape derivatives for x/y/z directions (tensor product element)
    auto ViewDShape() const { return Reshape(dxshape.Read(), ngll, ngll); }
    auto ViewDShapeW() const { return Reshape(dxshape_w.Read(), ngll, ngll); }

    /// Quadrature weight view [ngll]
    /// Note: Same GLL weights for x/y/z directions (tensor product element)
    auto ViewWgll() const { return Reshape(wgll.Read(), ngll); }

    /// Packed inverse Jacobian + detJ view [10, ngll, ngll, ngll, ne]
    /// Access: invJ_packed(comp, ix, iy, iz, e)
    /// Components: 0=j00, 1=j01, 2=j02, 3=j10, 4=j11, 5=j12, 6=j20, 7=j21, 8=j22, 9=detJ
    auto ViewInvJPacked() const { return Reshape(invJ_packed.Read(), 10, ngll, ngll, ngll, ne); }
};


/**
 * @brief Dimension-templated geometry selector
 *
 * Usage:
 *   SEMGeometry<2>::Type geom2d;  // SEMGeometry2D
 *   SEMGeometry<3>::Type geom3d;  // SEMGeometry3D
 */
template<int Dim>
struct SEMGeometry;

template<>
struct SEMGeometry<2> {
    using Type = SEMGeometry2D;
};

template<>
struct SEMGeometry<3> {
    using Type = SEMGeometry3D;
};


}  // namespace SEM

#endif  // SEM_GEOMETRY_HPP
