/**
 * @file MaterialField.hpp
 * @brief Storage classes for material properties at GLL points
 *
 * Contains:
 * - MaterialField: Flattened storage for 2D scalar fields
 * - MaterialField3D: Flattened storage for 3D scalar fields
 *
 * Memory Layout: All data stored in contiguous MFEM Vectors
 * - 2D: Index(e, j, i) = e * (nglly * ngllx) + j * ngllx + i
 * - 3D: Index(e, k, j, i) = e * (ngllz * nglly * ngllx) + k * (nglly * ngllx) + j * ngllx + i
 */

#ifndef SEM_MATERIAL_FIELD_HPP
#define SEM_MATERIAL_FIELD_HPP

#include <mfem.hpp>
#include "general/forall.hpp"  // MFEM 4.8 mfem.hpp drops this; needed for mfem::Reshape
#include "linalg/dtensor.hpp"  // For Reshape and DeviceTensor

namespace SEM {

using namespace mfem;

// =============================================================================
// MaterialField (2D)
// =============================================================================

/**
 * @class MaterialField
 * @brief Flattened storage for a scalar material property at GLL points
 *
 * Provides efficient contiguous memory layout with convenient accessors.
 * Can be converted to/from DeviceTensor views via MFEM's Reshape().
 */
class MaterialField {
public:
    /// Default constructor (empty field)
    MaterialField() = default;

    /// Construct with dimensions, allocates zero-initialized storage
    MaterialField(int ne, int ngllx, int nglly);

    /// Copy constructor
    MaterialField(const MaterialField& other);

    /// Move constructor
    MaterialField(MaterialField&& other) noexcept;

    /// Copy assignment
    MaterialField& operator=(const MaterialField& other);

    /// Move assignment
    MaterialField& operator=(MaterialField&& other) noexcept;

    /// Destructor
    ~MaterialField() = default;

    // =========================================================================
    // Size information
    // =========================================================================

    /// Number of elements
    int NumElements() const { return ne_; }

    /// Number of GLL points in x direction
    int NumGLLx() const { return ngllx_; }

    /// Number of GLL points in y direction
    int NumGLLy() const { return nglly_; }

    /// Total size of flattened array
    int Size() const { return data_.Size(); }

    /// Memory usage in bytes
    size_t MemoryUsage() const { return data_.Size() * sizeof(real_t); }

    // =========================================================================
    // Element access (for setup/debugging, not for hot loops)
    // =========================================================================

    /// Access element (e, j, i) - read/write
    real_t& operator()(int e, int j, int i) {
        return data_[e * stride_e_ + j * ngllx_ + i];
    }

    /// Access element (e, j, i) - read only
    real_t operator()(int e, int j, int i) const {
        return data_[e * stride_e_ + j * ngllx_ + i];
    }

    // =========================================================================
    // Bulk data access (for hot loops and MFEM operations)
    // =========================================================================

    /// Get underlying MFEM Vector (read/write)
    Vector& Data() { return data_; }

    /// Get underlying MFEM Vector (read only)
    const Vector& Data() const { return data_; }

    /// Get raw pointer (read/write)
    real_t* GetData() { return data_.GetData(); }

    /// Get raw pointer (read only)
    const real_t* GetData() const { return data_.GetData(); }

    /// Get read-only pointer (for MFEM device operations)
    const real_t* Read() const { return data_.Read(); }

    /// Get write pointer (for MFEM device operations)
    real_t* Write() { return data_.Write(); }

    /// Get read-write pointer (for MFEM device operations)
    real_t* ReadWrite() { return data_.ReadWrite(); }

    /// Get host write pointer (for CPU initialization code)
    real_t* HostWrite() { return data_.HostWrite(); }

    /// Get DeviceTensor view for efficient multi-dimensional access
    /// Layout: [ngllx, nglly, ne] (column-major for MFEM compatibility)
    auto View() const {
        return Reshape(data_.Read(), ngllx_, nglly_, ne_);
    }

    auto ViewWrite() {
        return Reshape(data_.Write(), ngllx_, nglly_, ne_);
    }

    /// Host-only view for reading (for CPU code like MassIntegrator)
    auto ViewHost() const {
        return Reshape(data_.HostRead(), ngllx_, nglly_, ne_);
    }

    /// Host-only view for writing (for CPU initialization code)
    auto ViewHostWrite() {
        return Reshape(data_.HostWrite(), ngllx_, nglly_, ne_);
    }

    // =========================================================================
    // Initialization methods
    // =========================================================================

    /// Initialize from MFEM Coefficient at GLL points
    void ProjectCoefficient(Coefficient& coef,
                            const ParFiniteElementSpace& fes,
                            const IntegrationRule& ir);

    /// Initialize from another MaterialField
    void CopyFrom(const MaterialField& other);

    /// Set all values to a constant
    void SetConstant(real_t value);

    /// Scale all values by a factor
    void Scale(real_t factor);

    // =========================================================================
    // Conversion to ParGridFunction
    // =========================================================================

    /**
     * @brief Convert MaterialField to ParGridFunction for visualization
     *
     * Maps GLL-point values to the corresponding DOFs in the finite element space.
     * Assumes the FE space uses GLL nodes (as in SEM with H1 elements).
     *
     * @param fes Finite element space (must match dimensions)
     * @param gf Output grid function (will be resized if needed)
     */
    void ToParGridFunction(const ParFiniteElementSpace& fes, ParGridFunction& gf) const;

    /**
     * @brief Create and return a new ParGridFunction from this field
     *
     * @param fes Finite element space
     * @return New ParGridFunction containing the field values (caller owns memory)
     */
    ParGridFunction* ToParGridFunction(ParFiniteElementSpace* fes) const;

    // =========================================================================
    // Conversion from ParGridFunction
    // =========================================================================

    /**
     * @brief Create MaterialField from a ParGridFunction
     *
     * Inverse of ToParGridFunction: maps DOF values back to GLL-point storage.
     *
     * @param fes Finite element space
     * @param gf Input grid function
     * @return MaterialField with GLL-point values
     */
    static MaterialField FromParGridFunction(const ParFiniteElementSpace& fes,
                                             const ParGridFunction& gf);

    // =========================================================================
    // Utility methods
    // =========================================================================

    /// Get minimum value across all points
    real_t Min() const;

    /// Get maximum value across all points
    real_t Max() const;

    /// Compute L2 norm
    real_t Norml2() const { return data_.Norml2(); }

    /// Print summary statistics
    void PrintStats(std::ostream& os = mfem::out) const;

private:
    Vector data_;           ///< Flattened storage [ne * nglly * ngllx]
    int ne_ = 0;            ///< Number of elements
    int ngllx_ = 0;         ///< GLL points in x
    int nglly_ = 0;         ///< GLL points in y
    int stride_e_ = 0;      ///< Stride between elements (= nglly * ngllx)
};


// =============================================================================
// MaterialField3D
// =============================================================================

/**
 * @class MaterialField3D
 * @brief Flattened storage for 3D scalar material property at GLL points
 */
class MaterialField3D {
public:
    MaterialField3D() = default;
    MaterialField3D(int ne, int ngllx, int nglly, int ngllz);

    int NumElements() const { return ne_; }
    int NumGLLx() const { return ngllx_; }
    int NumGLLy() const { return nglly_; }
    int NumGLLz() const { return ngllz_; }
    int Size() const { return data_.Size(); }
    size_t MemoryUsage() const { return data_.Size() * sizeof(real_t); }

    real_t& operator()(int e, int k, int j, int i) {
        return data_[e * stride_e_ + k * stride_k_ + j * ngllx_ + i];
    }

    real_t operator()(int e, int k, int j, int i) const {
        return data_[e * stride_e_ + k * stride_k_ + j * ngllx_ + i];
    }

    Vector& Data() { return data_; }
    const Vector& Data() const { return data_; }
    real_t* GetData() { return data_.GetData(); }
    const real_t* GetData() const { return data_.GetData(); }
    const real_t* Read() const { return data_.Read(); }
    real_t* Write() { return data_.Write(); }
    real_t* HostWrite() { return data_.HostWrite(); }

    /// Get DeviceTensor view for efficient multi-dimensional access
    /// Layout: [ngllx, nglly, ngllz, ne]
    auto View() const {
        return Reshape(data_.Read(), ngllx_, nglly_, ngllz_, ne_);
    }

    auto ViewWrite() {
        return Reshape(data_.Write(), ngllx_, nglly_, ngllz_, ne_);
    }

    /// Host-only view for reading (for CPU code like MassIntegrator)
    auto ViewHost() const {
        return Reshape(data_.HostRead(), ngllx_, nglly_, ngllz_, ne_);
    }

    /// Host-only view for writing (for CPU initialization code)
    auto ViewHostWrite() {
        return Reshape(data_.HostWrite(), ngllx_, nglly_, ngllz_, ne_);
    }

    void ProjectCoefficient(Coefficient& coef, const ParFiniteElementSpace& fes,
                            const IntegrationRule& ir);
    void SetConstant(real_t value);

    // =========================================================================
    // Conversion to ParGridFunction
    // =========================================================================

    /**
     * @brief Convert MaterialField3D to ParGridFunction for visualization
     *
     * Maps GLL-point values to the corresponding DOFs in the finite element space.
     * Assumes the FE space uses GLL nodes (as in SEM with H1 elements).
     *
     * @param fes Finite element space (must match dimensions)
     * @param gf Output grid function (will be resized if needed)
     */
    void ToParGridFunction(const ParFiniteElementSpace& fes, ParGridFunction& gf) const;

    /**
     * @brief Create and return a new ParGridFunction from this field
     *
     * @param fes Finite element space
     * @return New ParGridFunction containing the field values (caller owns memory)
     */
    ParGridFunction* ToParGridFunction(ParFiniteElementSpace* fes) const;

    /**
     * @brief Inverse of ToParGridFunction — build a 3D MaterialField from a
     * scalar ParGridFunction.
     */
    static MaterialField3D FromParGridFunction(const ParFiniteElementSpace& fes,
                                                const ParGridFunction& gf);

private:
    Vector data_;
    int ne_ = 0, ngllx_ = 0, nglly_ = 0, ngllz_ = 0;
    int stride_e_ = 0, stride_k_ = 0;
};

}  // namespace SEM

#endif  // SEM_MATERIAL_FIELD_HPP
