/**
 * @file SEMIsotropicElasticIntegrator.hpp
 * @brief Isotropic elastic stiffness integrator for SEM
 *
 * Direct implementation without template abstraction.
 * Stress computation: sigma = lambda * tr(eps) * I + 2 * mu * eps
 */

#ifndef SEM_ISOTROPIC_ELASTIC_INTEGRATOR_HPP
#define SEM_ISOTROPIC_ELASTIC_INTEGRATOR_HPP

#include <mfem.hpp>
#include "general/forall.hpp"  // MFEM 4.8 mfem.hpp drops this; needed for mfem::Reshape
#include "integ/core/SEMIntegratorBase.hpp"
#include "integ/core/SEMKernelDispatch.hpp"

namespace SEM {


using namespace mfem;

/**
 * @class SEMIsotropicElasticIntegrator2D
 * @brief 2D Isotropic elastic stiffness integrator
 */
class SEMIsotropicElasticIntegrator2D : public SEMIntegratorBase2D {
public:
    /**
     * @brief Constructor with separate lambda and mu vectors
     * @param lambda Lame's first parameter per GLL point
     * @param mu Shear modulus per GLL point
     */
    SEMIsotropicElasticIntegrator2D(const Vector& lambda, const Vector& mu) {
        int n = lambda.Size();
        material_params_.SetSize(2 * n);
        material_params_.UseDevice(true);
        real_t* data = material_params_.HostWrite();
        const real_t* lam = lambda.HostRead();
        const real_t* m = mu.HostRead();
        for (int i = 0; i < n; i++) {
            data[2*i + 0] = lam[i];
            data[2*i + 1] = m[i];
        }
        // Force host→device sync for GPU builds
        material_params_.Read();
    }

    ~SEMIsotropicElasticIntegrator2D() override = default;

    // Set members from base class and compute DOF ordering for vector field
    void AssemblePA(const FiniteElementSpace &fes) override {
        SEMIntegratorBase2D::AssemblePA(fes);
        dofs_.ComputeVector(fes, order_);
    }

    //For the reason of vectorization, using the macro dispatch here,
    //huge switch codes are generated for different orders....
    //This is the definition and declaration to call the AddMultPA_Opt kernel declared below.
    void AddMultPA(const Vector &x, Vector &y) const override {
        SEM_DISPATCH_NGLL(ngll_, AddMultPA_Opt, x, y);
    }

    const Vector& GetMaterialParams() const { return material_params_; }

    /// material_params_ is stored in comp->ix->iy->e order for GPU coalesced access
    /// This view is just a convenient wrapper to access the data using (comp, ix, iy, e) indexing,
    /// otherwise need to use like ic + 2*(ix + ngll*(iy + ngll*e)) to access the data in the kernel
    /// Component 0 = lambda, Component 1 = mu
    auto ViewMaterialParams() const {
        return Reshape(material_params_.Read(), 2, ngll_, ngll_, ne_);
    }

    size_t MemoryUsage() const override {
        return SEMIntegratorBase2D::MemoryUsage() +
               material_params_.Size() * sizeof(real_t);
    }

    // Kernel methods (public for CUDA extended lambda)
    // This will be called by AddMultPA with macro dispatch for different NGLL values, and the actual implementation will be in the .cpp file.
    template<int NGLL>
    void AddMultPA_Opt(const Vector &x, Vector &y) const;

protected:
    Vector material_params_;
};


/**
 * @class SEMIsotropicElasticIntegrator3D
 * @brief 3D Isotropic elastic stiffness integrator
 */
class SEMIsotropicElasticIntegrator3D : public SEMIntegratorBase3D {
public:
    /**
     * @brief Constructor with separate lambda and mu vectors
     * @param lambda Lame's first parameter per GLL point
     * @param mu Shear modulus per GLL point
     */
    SEMIsotropicElasticIntegrator3D(const Vector& lambda, const Vector& mu) {
        int n = lambda.Size();
        material_params_.SetSize(2 * n);
        material_params_.UseDevice(true);
        real_t* data = material_params_.HostWrite();
        const real_t* lam = lambda.HostRead();
        const real_t* m = mu.HostRead();
        for (int i = 0; i < n; i++) {
            data[2*i + 0] = lam[i];
            data[2*i + 1] = m[i];
        }
        // Force host→device sync for GPU builds
        material_params_.Read();
    }

    ~SEMIsotropicElasticIntegrator3D() override = default;

    // Set members from base class and compute DOF ordering for vector field
    void AssemblePA(const FiniteElementSpace &fes) override {
        SEMIntegratorBase3D::AssemblePA(fes);
        dofs_.ComputeVector(fes, order_);
    }

    //For the reason of vectorization, using the macro dispatch here,
    //huge switch codes are generated for different orders....
    //This is the definition and declaration to call the AddMultPA_Opt kernel declared below.
    void AddMultPA(const Vector &x, Vector &y) const override {
        SEM_DISPATCH_NGLL_3D(ngll_, AddMultPA_Opt, x, y);
    }

    const Vector& GetMaterialParams() const { return material_params_; }

    /// material_params_ is stored in comp->ix->iy->iz->e order for GPU coalesced access
    /// This view is just a convenient wrapper to access the data using (comp, ix, iy, iz, e) indexing,
    /// otherwise need to use like ic + 2*(ix + ngll*(iy + ngll*(iz + ngll*e))) to access the data in the kernel
    /// Component 0 = lambda, Component 1 = mu
    auto ViewMaterialParams() const {
        return Reshape(material_params_.Read(), 2, ngll_, ngll_, ngll_, ne_);
    }

    size_t MemoryUsage() const override {
        return SEMIntegratorBase3D::MemoryUsage() +
               material_params_.Size() * sizeof(real_t);
    }

    // Kernel methods (public for CUDA extended lambda)
    // This will be called by AddMultPA with macro dispatch for different NGLL values, and the actual implementation will be in the .cpp file.
    template<int NGLL>
    void AddMultPA_Opt(const Vector &x, Vector &y) const;

protected:
    Vector material_params_;
};


}  // namespace SEM

#endif  // SEM_ISOTROPIC_ELASTIC_INTEGRATOR_HPP
