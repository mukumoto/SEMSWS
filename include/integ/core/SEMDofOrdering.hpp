/**
 * @file SEMDofOrdering.hpp
 * @brief DOF (gather_map) ordering utilities for SEM integrators
 *
 * Provides functions to compute gather_map arrays that map local element-GLL indices
 * to global DOF indices. Supports both scalar and vector fields.
 */

#ifndef SEM_DOF_ORDERING_HPP
#define SEM_DOF_ORDERING_HPP

#include <mfem.hpp>
#include "general/forall.hpp"  // MFEM 4.8 mfem.hpp drops this; needed for mfem::Reshape

namespace SEM {


using namespace mfem;

/**
 * @brief Field type for DOF ordering
 */
enum class FieldType {
    Scalar,  // 1 component (acoustic pressure)
    Vector   // DIM components (elastic displacement)
};

/**
 * @brief 2D DOF ordering data
 *
 * For scalar fields: gather_map maps (e, iy, ix) -> global DOF
 * For vector fields: gather_map_x, gather_map_y map (e, iy, ix) -> component DOF
 */
struct SEMDofOrdering2D {
    // Scalar field mapping
    Array<int> gather_map;

    // Vector field mapping (2 components)
    Array<int> gather_map_x;
    Array<int> gather_map_y;

    // State
    int ne = 0;
    int ngll = 0;
    FieldType field_type = FieldType::Vector;

    // Cached device pointers (set during SyncToDevice, use d_ prefix)
    const int* d_gather_map = nullptr;
    const int* d_gather_map_x = nullptr;
    const int* d_gather_map_y = nullptr;

    /// Check if DOF ordering is valid
    bool IsValid() const { return gather_map.Size() > 0 || gather_map_x.Size() > 0; }

    /// Compute DOF ordering for scalar field
    void ComputeScalar(const FiniteElementSpace& fes, int order);

    /// Compute DOF ordering for vector field
    void ComputeVector(const FiniteElementSpace& fes, int order);

    /// Memory usage in bytes
    size_t MemoryUsage() const {
        return (gather_map.Size() + gather_map_x.Size() + gather_map_y.Size()) * sizeof(int);
    }

    /// Enable GPU memory for all arrays
    void EnableDevice() {
        gather_map.GetMemory().UseDevice(true);
        gather_map_x.GetMemory().UseDevice(true);
        gather_map_y.GetMemory().UseDevice(true);
    }

    /// Force host→device sync and cache device pointers (call after EnableDevice)
    void SyncToDevice() {
        if (field_type == FieldType::Scalar) {
            d_gather_map = gather_map.Read();
        } else {
            d_gather_map_x = gather_map_x.Read();
            d_gather_map_y = gather_map_y.Read();
        }
    }

    /// Check if device pointers are cached
    bool IsDeviceReady() const {
        if (field_type == FieldType::Scalar) {
            return d_gather_map != nullptr;
        }
        return d_gather_map_x != nullptr;
    }

    // =========================================================================
    // View Factory Methods (Zero-Overhead MFEM-style access)
    // =========================================================================

    /// Scalar gather_map view [ngll, ngll, ne] - access: gather_map(ix, iy, e)
    auto ViewGatherMap() const { return Reshape(gather_map.Read(), ngll, ngll, ne); }

    /// Vector gather_map views [ngll, ngll, ne] - access: gather_map_x(ix, iy, e)
    auto ViewGatherMapX() const { return Reshape(gather_map_x.Read(), ngll, ngll, ne); }
    auto ViewGatherMapY() const { return Reshape(gather_map_y.Read(), ngll, ngll, ne); }
};


/**
 * @brief 3D DOF ordering data
 *
 * For scalar fields: gather_map maps (e, iz, iy, ix) -> global DOF
 * For vector fields: gather_map_x, gather_map_y, gather_map_z map (e, iz, iy, ix) -> component DOF
 */
struct SEMDofOrdering3D {
    // Scalar field mapping
    Array<int> gather_map;

    // Vector field mapping (3 components)
    Array<int> gather_map_x;
    Array<int> gather_map_y;
    Array<int> gather_map_z;

    // State
    int ne = 0;
    int ngll = 0;
    FieldType field_type = FieldType::Vector;

    // Cached device pointers (set during SyncToDevice, use d_ prefix)
    const int* d_gather_map = nullptr;
    const int* d_gather_map_x = nullptr;
    const int* d_gather_map_y = nullptr;
    const int* d_gather_map_z = nullptr;

    /// Check if DOF ordering is valid
    bool IsValid() const { return gather_map.Size() > 0 || gather_map_x.Size() > 0; }

    /// Compute DOF ordering for scalar field
    void ComputeScalar(const FiniteElementSpace& fes, int order);

    /// Compute DOF ordering for vector field
    void ComputeVector(const FiniteElementSpace& fes, int order);

    /// Memory usage in bytes
    size_t MemoryUsage() const {
        return (gather_map.Size() + gather_map_x.Size() + gather_map_y.Size() + gather_map_z.Size()) * sizeof(int);
    }

    /// Enable GPU memory for all arrays
    void EnableDevice() {
        gather_map.GetMemory().UseDevice(true);
        gather_map_x.GetMemory().UseDevice(true);
        gather_map_y.GetMemory().UseDevice(true);
        gather_map_z.GetMemory().UseDevice(true);
    }

    /// Force host→device sync and cache device pointers (call after EnableDevice)
    void SyncToDevice() {
        if (field_type == FieldType::Scalar) {
            d_gather_map = gather_map.Read();
        } else {
            d_gather_map_x = gather_map_x.Read();
            d_gather_map_y = gather_map_y.Read();
            d_gather_map_z = gather_map_z.Read();
        }
    }

    /// Check if device pointers are cached
    bool IsDeviceReady() const {
        if (field_type == FieldType::Scalar) {
            return d_gather_map != nullptr;
        }
        return d_gather_map_x != nullptr;
    }

    // =========================================================================
    // View Factory Methods (Zero-Overhead MFEM-style access)
    // =========================================================================

    /// Scalar gather_map view [ngll, ngll, ngll, ne] - access: gather_map(ix, iy, iz, e)
    auto ViewGatherMap() const { return Reshape(gather_map.Read(), ngll, ngll, ngll, ne); }

    /// Vector gather_map views [ngll, ngll, ngll, ne] - access: gather_map_x(ix, iy, iz, e)
    auto ViewGatherMapX() const { return Reshape(gather_map_x.Read(), ngll, ngll, ngll, ne); }
    auto ViewGatherMapY() const { return Reshape(gather_map_y.Read(), ngll, ngll, ngll, ne); }
    auto ViewGatherMapZ() const { return Reshape(gather_map_z.Read(), ngll, ngll, ngll, ne); }
};


/**
 * @brief Dimension-templated DOF ordering selector
 */
template<int Dim>
struct SEMDofOrdering;

template<>
struct SEMDofOrdering<2> {
    using Type = SEMDofOrdering2D;
};

template<>
struct SEMDofOrdering<3> {
    using Type = SEMDofOrdering3D;
};


}  // namespace SEM

#endif  // SEM_DOF_ORDERING_HPP
