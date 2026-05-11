/**
 * @file SEMIsotropicAcousticIntegrator.hpp
 * @brief Isotropic acoustic stiffness integrator for SEM
 *
 * Direct implementation without template abstraction.
 * Flux computation: q = (1/rho) * grad(p)
 */

#ifndef SEM_ISOTROPIC_ACOUSTIC_INTEGRATOR_HPP
#define SEM_ISOTROPIC_ACOUSTIC_INTEGRATOR_HPP

#include <mfem.hpp>
#include "general/forall.hpp"  // MFEM 4.8 mfem.hpp drops this; needed for mfem::Reshape
#include "integ/core/SEMIntegratorBase.hpp"
#include "integ/core/SEMKernelDispatch.hpp"

namespace SEM {


using namespace mfem;

/**
 * @class SEMIsotropicAcousticIntegrator2D
 * @brief 2D Isotropic acoustic stiffness integrator
 */
class SEMIsotropicAcousticIntegrator2D : public SEMIntegratorBase2D {
public:
    /**
     * @brief Constructor with inverse density
     * @param inv_rho Inverse density (1/rho) per GLL point
     */
    SEMIsotropicAcousticIntegrator2D(const Vector& inv_rho) : material_params_(inv_rho) {
        material_params_.UseDevice(true);
        // Force host→device sync for GPU builds
        material_params_.Read();
    }

    ~SEMIsotropicAcousticIntegrator2D() override = default;

    // Set members from base class and compute DOF ordering for scalar field
    void AssemblePA(const FiniteElementSpace &fes) override {
        SEMIntegratorBase2D::AssemblePA(fes);
        dofs_.ComputeScalar(fes, order_);
    }

    //For the reason of vectorization, using the macro dispatch here,
    //huge switch codes are generated for different orders....
    //This is the definition and declaration to call the AddMultPA_Opt kernel declared below.
    void AddMultPA(const Vector &x, Vector &y) const override {
        SEM_DISPATCH_NGLL(ngll_, AddMultPA_Opt, x, y);
    }

    const Vector& GetMaterialParams() const { return material_params_; }

    /// material_params_ is stored in ix->iy->e order for GPU coalesced access
    /// This view is just a convenient wrapper to access the data using (ix, iy, e) indexing,
    /// otherwise need to use like ix + ngll*(iy + ngll*e) to access the data in the kernel
    /// Single component: inv_rho (1/ρ)
    auto ViewInvRho() const { return Reshape(material_params_.Read(), ngll_, ngll_, ne_); }

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
 * @class SEMIsotropicAcousticIntegrator3D
 * @brief 3D Isotropic acoustic stiffness integrator
 */
class SEMIsotropicAcousticIntegrator3D : public SEMIntegratorBase3D {
public:
    /**
     * @brief Constructor with inverse density
     * @param inv_rho Inverse density (1/rho) per GLL point
     */
    SEMIsotropicAcousticIntegrator3D(const Vector& inv_rho) : material_params_(inv_rho) {
        material_params_.UseDevice(true);
        // Force host→device sync for GPU builds
        material_params_.Read();
    }

    ~SEMIsotropicAcousticIntegrator3D() override = default;

    // Set members from base class and compute DOF ordering for scalar field
    void AssemblePA(const FiniteElementSpace &fes) override {
        SEMIntegratorBase3D::AssemblePA(fes);
        dofs_.ComputeScalar(fes, order_);
    }

    //For the reason of vectorization, using the macro dispatch here,
    //huge switch codes are generated for different orders....
    //This is the definition and declaration to call the AddMultPA_Opt kernel declared below.
    void AddMultPA(const Vector &x, Vector &y) const override {
        SEM_DISPATCH_NGLL_3D(ngll_, AddMultPA_Opt, x, y);
    }

    const Vector& GetMaterialParams() const { return material_params_; }

    /// material_params_ is stored in ix->iy->iz->e order for GPU coalesced access
    /// This view is just a convenient wrapper to access the data using (ix, iy, iz, e) indexing,
    /// otherwise need to use like ix + ngll*(iy + ngll*(iz + ngll*e)) to access the data in the kernel
    /// Single component: inv_rho (1/ρ)
    auto ViewInvRho() const { return Reshape(material_params_.Read(), ngll_, ngll_, ngll_, ne_); }

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

#endif  // SEM_ISOTROPIC_ACOUSTIC_INTEGRATOR_HPP
