/**
 * @file SEMIntegratorBase.hpp
 * @brief Dimension-templated base class for SEM integrators
 *
 * Provides the common base functionality for all SEM stiffness and mass integrators.
 * Uses SEMGeometry and SEMDofOrdering for geometry and DOF management.
 */

#ifndef SEM_INTEGRATOR_BASE_HPP
#define SEM_INTEGRATOR_BASE_HPP

#include <mfem.hpp>
#include "integ/core/SEMGeometry.hpp"
#include "integ/core/SEMDofOrdering.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "util/FESOrder.hpp"

namespace SEM {


using namespace mfem;

/**
 * @brief Base class for 2D SEM wave integrators
 *
 * Stores geometry and DOF ordering for 2D SEM integration.
 * Derived classes implement physics-specific kernels.
 */
class SEMIntegratorBase2D : public BilinearFormIntegrator {
protected:
    // Finite element space reference
    const FiniteElementSpace* fespace_ = nullptr;

    // State
    int order_ = 0;
    int ngll_ = 0;
    int ne_ = 0;

    // Geometry data
    SEMGeometry2D geom_;

    // DOF ordering (derived class chooses scalar or vector)
    SEMDofOrdering2D dofs_;

public:
    SEMIntegratorBase2D() = default;
    virtual ~SEMIntegratorBase2D() = default;

    /**
     * @brief Called once during BilinearForm::Assemble()
     *
     * Computes geometry if not already done. Derived classes override
     * to also compute DOF ordering and material projection.
     */
    void AssemblePA(const FiniteElementSpace &fes) override {
        fespace_ = &fes;

        Mesh* mesh = fes.GetMesh();
        ne_ = mesh->GetNE();
        order_ = SafeFESOrder(fes);
        ngll_ = order_ + 1;

        if (!geom_.IsValid()) {
            geom_.Compute(fes);
        }
    }

    /**
     * @brief Apply operator action: y += K*x
     *
     * Must be implemented by derived classes.
     */
    void AddMultPA(const Vector &x, Vector &y) const override = 0;

    /**
     * @brief Transpose is same as forward for symmetric operators
     */
    void AddMultTransposePA(const Vector &x, Vector &y) const override {
        AddMultPA(x, y);
    }

    /// Memory usage in bytes
    virtual size_t MemoryUsage() const {
        return geom_.MemoryUsage() + dofs_.MemoryUsage();
    }

    // =========================================================================
    // Attenuation interface (for viscoelastic integrators)
    // =========================================================================

    /// Check if this integrator has an attenuation model
    virtual bool HasAttenuationModel() const { return false; }

    /// Reset attenuation state (no-op for non-viscoelastic integrators)
    virtual void ResetAttenuationState() {}

    // =========================================================================
    // Device Initialization
    // =========================================================================

    /// Initialize device memory for geometry and DOF data
    virtual void DeviceInit() {
        geom_.EnableDevice();
        geom_.SyncToDevice();
        dofs_.EnableDevice();
        dofs_.SyncToDevice();
    }

    // Accessors
    int Order() const { return order_; }
    int NumGLL() const { return ngll_; }
    int NumElements() const { return ne_; }
    const SEMGeometry2D& Geometry() const { return geom_; }
    const SEMDofOrdering2D& Dofs() const { return dofs_; }
};


/**
 * @brief Base class for 3D SEM wave integrators
 *
 * Stores geometry and DOF ordering for 3D SEM integration.
 * Derived classes implement physics-specific kernels.
 */
class SEMIntegratorBase3D : public BilinearFormIntegrator {
protected:
    // Finite element space reference
    const FiniteElementSpace* fespace_ = nullptr;

    // State
    int order_ = 0;
    int ngll_ = 0;
    int ne_ = 0;

    // Geometry data
    SEMGeometry3D geom_;

    // DOF ordering (derived class chooses scalar or vector)
    SEMDofOrdering3D dofs_;

public:
    SEMIntegratorBase3D() = default;
    virtual ~SEMIntegratorBase3D() = default;

    /**
     * @brief Called once during BilinearForm::Assemble()
     *
     * Computes geometry if not already done. Derived classes override
     * to also compute DOF ordering and material projection.
     */
    void AssemblePA(const FiniteElementSpace &fes) override {
        fespace_ = &fes;

        Mesh* mesh = fes.GetMesh();
        ne_ = mesh->GetNE();
        order_ = SafeFESOrder(fes);
        ngll_ = order_ + 1;

        if (!geom_.IsValid()) {
            geom_.Compute(fes);
        }
    }

    /**
     * @brief Apply operator action: y += K*x
     *
     * Must be implemented by derived classes.
     */
    void AddMultPA(const Vector &x, Vector &y) const override = 0;

    /**
     * @brief Transpose is same as forward for symmetric operators
     */
    void AddMultTransposePA(const Vector &x, Vector &y) const override {
        AddMultPA(x, y);
    }

    /// Memory usage in bytes
    virtual size_t MemoryUsage() const {
        return geom_.MemoryUsage() + dofs_.MemoryUsage();
    }

    // =========================================================================
    // Attenuation interface (for viscoelastic integrators)
    // =========================================================================

    /// Check if this integrator has an attenuation model
    virtual bool HasAttenuationModel() const { return false; }

    /// Reset attenuation state (no-op for non-viscoelastic integrators)
    virtual void ResetAttenuationState() {}

    // =========================================================================
    // Device Initialization
    // =========================================================================

    /// Initialize device memory for geometry and DOF data
    virtual void DeviceInit() {
        geom_.EnableDevice();
        geom_.SyncToDevice();
        dofs_.EnableDevice();
        dofs_.SyncToDevice();
    }

    // Accessors
    int Order() const { return order_; }
    int NumGLL() const { return ngll_; }
    int NumElements() const { return ne_; }
    const SEMGeometry3D& Geometry() const { return geom_; }
    const SEMDofOrdering3D& Dofs() const { return dofs_; }
};


/**
 * @brief Dimension-templated integrator base selector
 */
template<int Dim>
struct SEMIntegratorBase;

template<>
struct SEMIntegratorBase<2> {
    using Type = SEMIntegratorBase2D;
};

template<>
struct SEMIntegratorBase<3> {
    using Type = SEMIntegratorBase3D;
};


}  // namespace SEM

#endif  // SEM_INTEGRATOR_BASE_HPP
