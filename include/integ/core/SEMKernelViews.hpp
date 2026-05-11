/**
 * @file SEMKernelViews.hpp
 * @brief Zero-overhead view helpers for SEM kernels
 *
 * Provides pre-created DeviceTensor views for kernel access following
 * MFEM and libCEED patterns. All views are created BEFORE forall launch
 * and captured by value for zero-overhead GPU access.
 *
 * Memory layout: (local_indices..., element) - Element is SLOWEST
 * This matches both libCEED and MFEM conventions:
 *   - libCEED: strides_elem = NGLL^3 (element stride is largest)
 *   - MFEM: Reshape(data, Q1D, Q1D, Q1D, NE) (NE last = slowest)
 *
 * Example usage:
 *   auto views = KernelViews3D<NGLL>::Create(geom, dofs, ne);
 *   mfem::forall(ne, [=] MFEM_HOST_DEVICE (int e) {
 *       pot[ix][iy][iz] = data[views.gather_map(ix, iy, iz, e)];
 *       grad += pot[k][iy][iz] * views.dxshape(ix, k);
 *   });
 */

#ifndef SEM_KERNEL_VIEWS_HPP
#define SEM_KERNEL_VIEWS_HPP

#include <mfem.hpp>
#include "general/forall.hpp"  // MFEM 4.8 mfem.hpp drops this; needed for mfem::Reshape
#include "integ/core/SEMGeometry.hpp"
#include "integ/core/SEMDofOrdering.hpp"

namespace SEM {


using namespace mfem;

// =============================================================================
// 2D Kernel Views
// =============================================================================

/**
 * @brief Pre-created views for 2D scalar kernel access
 *
 * All views created BEFORE forall launch, captured by value.
 * Zero overhead: DeviceTensor is stack-allocated, index calc inlined.
 */
template<int NGLL>
struct ScalarKernelViews2D {
    // DOF ordering [ngll, ngll, ne]
    DeviceTensor<3, const int> gather_map;

    // Shape derivatives [ngll, ngll]
    DeviceTensor<2, const real_t> dxshape;
    DeviceTensor<2, const real_t> dyshape;
    DeviceTensor<2, const real_t> dxshape_w;
    DeviceTensor<2, const real_t> dyshape_w;

    // Quadrature weights [ngll]
    DeviceTensor<1, const real_t> wx;
    DeviceTensor<1, const real_t> wy;

    // Factory method
    static ScalarKernelViews2D Create(
        const SEMGeometry2D& geom,
        const SEMDofOrdering2D& dofs,
        int ne)
    {
        ScalarKernelViews2D v;
        v.gather_map = Reshape(dofs.gather_map.Read(), NGLL, NGLL, ne);

        v.dxshape = Reshape(geom.dxshape.Read(), NGLL, NGLL);
        v.dyshape = Reshape(geom.dyshape.Read(), NGLL, NGLL);
        v.dxshape_w = Reshape(geom.dxshape_w.Read(), NGLL, NGLL);
        v.dyshape_w = Reshape(geom.dyshape_w.Read(), NGLL, NGLL);

        v.wx = Reshape(geom.wx.Read(), NGLL);
        v.wy = Reshape(geom.wy.Read(), NGLL);

        return v;
    }
};

/**
 * @brief Pre-created views for 2D vector kernel access
 */
template<int NGLL>
struct VectorKernelViews2D {
    // DOF ordering [ngll, ngll, ne]
    DeviceTensor<3, const int> gather_map_x;
    DeviceTensor<3, const int> gather_map_y;

    // Shape derivatives [ngll, ngll]
    DeviceTensor<2, const real_t> dxshape;
    DeviceTensor<2, const real_t> dyshape;
    DeviceTensor<2, const real_t> dxshape_w;
    DeviceTensor<2, const real_t> dyshape_w;

    // Quadrature weights [ngll]
    DeviceTensor<1, const real_t> wx;
    DeviceTensor<1, const real_t> wy;

    // Factory method
    static VectorKernelViews2D Create(
        const SEMGeometry2D& geom,
        const SEMDofOrdering2D& dofs,
        int ne)
    {
        VectorKernelViews2D v;
        v.gather_map_x = Reshape(dofs.gather_map_x.Read(), NGLL, NGLL, ne);
        v.gather_map_y = Reshape(dofs.gather_map_y.Read(), NGLL, NGLL, ne);

        v.dxshape = Reshape(geom.dxshape.Read(), NGLL, NGLL);
        v.dyshape = Reshape(geom.dyshape.Read(), NGLL, NGLL);
        v.dxshape_w = Reshape(geom.dxshape_w.Read(), NGLL, NGLL);
        v.dyshape_w = Reshape(geom.dyshape_w.Read(), NGLL, NGLL);

        v.wx = Reshape(geom.wx.Read(), NGLL);
        v.wy = Reshape(geom.wy.Read(), NGLL);

        return v;
    }
};

// =============================================================================
// 3D Kernel Views
// =============================================================================

/**
 * @brief Pre-created views for 3D scalar kernel access
 */
template<int NGLL>
struct ScalarKernelViews3D {
    // DOF ordering [ngll, ngll, ngll, ne]
    DeviceTensor<4, const int> gather_map;

    // Shape derivatives [ngll, ngll]
    DeviceTensor<2, const real_t> dxshape;
    DeviceTensor<2, const real_t> dyshape;
    DeviceTensor<2, const real_t> dzshape;
    DeviceTensor<2, const real_t> dxshape_w;
    DeviceTensor<2, const real_t> dyshape_w;
    DeviceTensor<2, const real_t> dzshape_w;

    // Quadrature weights [ngll]
    DeviceTensor<1, const real_t> wx;
    DeviceTensor<1, const real_t> wy;
    DeviceTensor<1, const real_t> wz;

    // Factory method
    static ScalarKernelViews3D Create(
        const SEMGeometry3D& geom,
        const SEMDofOrdering3D& dofs,
        int ne)
    {
        ScalarKernelViews3D v;
        v.gather_map = Reshape(dofs.gather_map.Read(), NGLL, NGLL, NGLL, ne);

        v.dxshape = Reshape(geom.dxshape.Read(), NGLL, NGLL);
        v.dyshape = Reshape(geom.dyshape.Read(), NGLL, NGLL);
        v.dzshape = Reshape(geom.dzshape.Read(), NGLL, NGLL);
        v.dxshape_w = Reshape(geom.dxshape_w.Read(), NGLL, NGLL);
        v.dyshape_w = Reshape(geom.dyshape_w.Read(), NGLL, NGLL);
        v.dzshape_w = Reshape(geom.dzshape_w.Read(), NGLL, NGLL);

        v.wx = Reshape(geom.wx.Read(), NGLL);
        v.wy = Reshape(geom.wy.Read(), NGLL);
        v.wz = Reshape(geom.wz.Read(), NGLL);

        return v;
    }
};

/**
 * @brief Pre-created views for 3D vector kernel access
 */
template<int NGLL>
struct VectorKernelViews3D {
    // DOF ordering [ngll, ngll, ngll, ne]
    DeviceTensor<4, const int> gather_map_x;
    DeviceTensor<4, const int> gather_map_y;
    DeviceTensor<4, const int> gather_map_z;

    // Shape derivatives [ngll, ngll]
    DeviceTensor<2, const real_t> dxshape;
    DeviceTensor<2, const real_t> dyshape;
    DeviceTensor<2, const real_t> dzshape;
    DeviceTensor<2, const real_t> dxshape_w;
    DeviceTensor<2, const real_t> dyshape_w;
    DeviceTensor<2, const real_t> dzshape_w;

    // Quadrature weights [ngll]
    DeviceTensor<1, const real_t> wx;
    DeviceTensor<1, const real_t> wy;
    DeviceTensor<1, const real_t> wz;

    // Factory method
    static VectorKernelViews3D Create(
        const SEMGeometry3D& geom,
        const SEMDofOrdering3D& dofs,
        int ne)
    {
        VectorKernelViews3D v;
        v.gather_map_x = Reshape(dofs.gather_map_x.Read(), NGLL, NGLL, NGLL, ne);
        v.gather_map_y = Reshape(dofs.gather_map_y.Read(), NGLL, NGLL, NGLL, ne);
        v.gather_map_z = Reshape(dofs.gather_map_z.Read(), NGLL, NGLL, NGLL, ne);

        v.dxshape = Reshape(geom.dxshape.Read(), NGLL, NGLL);
        v.dyshape = Reshape(geom.dyshape.Read(), NGLL, NGLL);
        v.dzshape = Reshape(geom.dzshape.Read(), NGLL, NGLL);
        v.dxshape_w = Reshape(geom.dxshape_w.Read(), NGLL, NGLL);
        v.dyshape_w = Reshape(geom.dyshape_w.Read(), NGLL, NGLL);
        v.dzshape_w = Reshape(geom.dzshape_w.Read(), NGLL, NGLL);

        v.wx = Reshape(geom.wx.Read(), NGLL);
        v.wy = Reshape(geom.wy.Read(), NGLL);
        v.wz = Reshape(geom.wz.Read(), NGLL);

        return v;
    }
};

// =============================================================================
// Inverse Jacobian View Helpers
// =============================================================================

/**
 * @brief 2D Inverse Jacobian access helper (AoS layout)
 *
 * Provides matrix-like access to AoS inverse Jacobian data.
 */
struct InvJacobianView2D {
    const real_t* data;  // [ne * ngll^2 * 4] in AoS: [j00, j11, j01, j10]
    int ngll2;           // ngll^2

    MFEM_HOST_DEVICE inline
    void Get(int local_idx, int e,
             real_t& j00, real_t& j11, real_t& j01, real_t& j10) const {
        int base = (e * ngll2 + local_idx) * 4;
        j00 = data[base + 0];
        j11 = data[base + 1];
        j01 = data[base + 2];
        j10 = data[base + 3];
    }

    static InvJacobianView2D Create(const SEMGeometry2D& geom) {
        return { geom.invJ.Read(), geom.ngll * geom.ngll };
    }
};

/**
 * @brief 3D Inverse Jacobian access helper (SoA layout)
 *
 * Keeps SoA storage (for vectorization), provides matrix-like access.
 * All 9 pointers are captured by value into GPU registers.
 */
struct InvJacobianView3D {
    const real_t* j00; const real_t* j01; const real_t* j02;
    const real_t* j10; const real_t* j11; const real_t* j12;
    const real_t* j20; const real_t* j21; const real_t* j22;

    /// Matrix element access: invJ(row, col, global_idx)
    MFEM_HOST_DEVICE inline
    real_t operator()(int row, int col, int idx) const {
        // This array access compiles to a single pointer read when
        // row and col are compile-time constants
        const real_t* ptrs[9] = {j00, j01, j02, j10, j11, j12, j20, j21, j22};
        return ptrs[row * 3 + col][idx];
    }

    /// Get a row of the inverse Jacobian (for gradient transformation)
    MFEM_HOST_DEVICE inline
    void GetRow(int row, int idx, real_t& c0, real_t& c1, real_t& c2) const {
        const real_t* ptrs[9] = {j00, j01, j02, j10, j11, j12, j20, j21, j22};
        c0 = ptrs[row * 3 + 0][idx];
        c1 = ptrs[row * 3 + 1][idx];
        c2 = ptrs[row * 3 + 2][idx];
    }

    static InvJacobianView3D Create(const SEMGeometry3D& geom) {
        return {
            geom.invJ_00.Read(), geom.invJ_01.Read(), geom.invJ_02.Read(),
            geom.invJ_10.Read(), geom.invJ_11.Read(), geom.invJ_12.Read(),
            geom.invJ_20.Read(), geom.invJ_21.Read(), geom.invJ_22.Read()
        };
    }
};

}  // namespace SEM

#endif  // SEM_KERNEL_VIEWS_HPP
