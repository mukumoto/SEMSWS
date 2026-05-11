/**
 * @file SEMVisco_IsotropicElasticIntegrator.hpp
 * @brief Viscoelastic (isotropic elastic with attenuation) stiffness integrator for SEM
 *
 * Isotropic elastic with Generalized Zener (SLS) attenuation.
 * Direct implementation without template abstraction.
 */

#ifndef SEM_VISCO_ISOTROPIC_ELASTIC_INTEGRATOR_HPP
#define SEM_VISCO_ISOTROPIC_ELASTIC_INTEGRATOR_HPP

#include <mfem.hpp>
#include "general/forall.hpp"  // MFEM 4.8 mfem.hpp drops this; needed for mfem::Reshape
#include "integ/core/SEMIntegratorBase.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "integ/attenuation/GeneralizedZener.hpp"

namespace SEM {


using namespace mfem;

/**
 * @class SEMVisco_IsotropicElasticIntegrator2D
 * @brief 2D Viscoelastic stiffness integrator (Generalized Zener attenuation)
 */
class SEMVisco_IsotropicElasticIntegrator2D : public SEMIntegratorBase2D {
public:
    /**
     * @brief Constructor with separate kappa and mu vectors
     * @param kappa Bulk modulus per GLL point (copied, will be modified with correction)
     * @param mu Shear modulus per GLL point (copied, will be modified with correction)
     *
     * The vectors are copied internally and will be modified by EnableAttenuation()
     * to apply the unrelaxed modulus correction.
     */
    SEMVisco_IsotropicElasticIntegrator2D(const Vector& kappa, const Vector& mu)
        : kappa_vec_(kappa), mu_vec_(mu)
    {
        int n = kappa.Size();
        material_params_.SetSize(2 * n);
        material_params_.UseDevice(true);
        // Don't copy yet - will be copied after EnableAttenuation applies correction
    }

    /// Copy corrected kappa/mu to material_params_ (call after EnableAttenuation)
    void FinalizeMaterialParams() {
        int n = kappa_vec_.Size();
        real_t* data = material_params_.HostWrite();
        const real_t* k = kappa_vec_.HostRead();
        const real_t* m = mu_vec_.HostRead();
        for (int i = 0; i < n; i++) {
            data[2*i + 0] = k[i];  // kappa (corrected)
            data[2*i + 1] = m[i];  // mu (corrected)
        }
        // Force host→device sync for GPU builds
        material_params_.Read();
    }

    ~SEMVisco_IsotropicElasticIntegrator2D() override = default;

    // Set members from base class and compute DOF ordering for vector field
    void AssemblePA(const FiniteElementSpace &fes) override {
        SEMIntegratorBase2D::AssemblePA(fes);
        dofs_.ComputeVector(fes, order_);
    }

    //For the reason of vectorization, using the macro dispatch here,
    //huge switch codes are generated for different orders....
    //This is the definition and declaration to call the AddMultPA_Visco_Opt kernel declared below.
    void AddMultPA(const Vector &x, Vector &y) const override {
        SEM_DISPATCH_NGLL(ngll_, AddMultPA_Visco_Opt, x, y);
    }

    /// Get attenuation model
    GeneralizedZener2D& GetAttenuation() { return attenuation_; }
    const GeneralizedZener2D& GetAttenuation() const { return attenuation_; }

    /// Check if attenuation is enabled
    bool HasAttenuation() const { return attenuation_.HasMemory(); }

    // =========================================================================
    // Attenuation interface (override from base class)
    // =========================================================================

    bool HasAttenuationModel() const override { return HasAttenuation(); }

    void ResetAttenuationState() override { attenuation_.Reset(); }

    const Vector& GetMaterialParams() const { return material_params_; }

    /// material_params_ is stored in comp->ix->iy->e order for GPU coalesced access
    /// This view is just a convenient wrapper to access the data using (comp, ix, iy, e) indexing,
    /// otherwise need to use like ic + 2*(ix + ngll*(iy + ngll*e)) to access the data in the kernel
    /// Component 0 = kappa (bulk modulus, corrected), Component 1 = mu (shear modulus, corrected)
    auto ViewMaterialParams() const {
        return Reshape(material_params_.Read(), 2, ngll_, ngll_, ne_);
    }

    size_t MemoryUsage() const override {
        return SEMIntegratorBase2D::MemoryUsage() +
               material_params_.Size() * sizeof(real_t) +
               attenuation_.MemoryUsage();
    }

    // Kernel methods (public for CUDA extended lambda)
    // This will be called by AddMultPA with macro dispatch for different NGLL values, and the actual implementation will be in the .cpp file.
    template<int NGLL>
    void AddMultPA_Visco_Opt(const Vector &x, Vector &y) const;

    /// Get kappa/mu vectors (for passing to EnableAttenuation)
    Vector& KappaVec() { return kappa_vec_; }
    Vector& MuVec() { return mu_vec_; }

protected:
    Vector material_params_;
    GeneralizedZener2D attenuation_;
    Vector kappa_vec_;  ///< Kappa vector (modified by EnableAttenuation)
    Vector mu_vec_;     ///< Mu vector (modified by EnableAttenuation)
};


/**
 * @class SEMVisco_IsotropicElasticIntegrator3D
 * @brief 3D Viscoelastic stiffness integrator (Generalized Zener attenuation)
 */
class SEMVisco_IsotropicElasticIntegrator3D : public SEMIntegratorBase3D {
public:
    /**
     * @brief Constructor with separate kappa and mu vectors
     * @param kappa Bulk modulus per GLL point (copied, will be modified with correction)
     * @param mu Shear modulus per GLL point (copied, will be modified with correction)
     *
     * The vectors are copied internally and will be modified by EnableAttenuation()
     * to apply the unrelaxed modulus correction.
     */
    SEMVisco_IsotropicElasticIntegrator3D(const Vector& kappa, const Vector& mu)
        : kappa_vec_(kappa), mu_vec_(mu)
    {
        int n = kappa.Size();
        material_params_.SetSize(2 * n);
        material_params_.UseDevice(true);
        // Don't copy yet - will be copied after EnableAttenuation applies correction
    }

    /// Copy corrected kappa/mu to material_params_ (call after EnableAttenuation)
    void FinalizeMaterialParams() {
        int n = kappa_vec_.Size();
        real_t* data = material_params_.HostWrite();
        const real_t* k = kappa_vec_.HostRead();
        const real_t* m = mu_vec_.HostRead();
        for (int i = 0; i < n; i++) {
            data[2*i + 0] = k[i];  // kappa (corrected)
            data[2*i + 1] = m[i];  // mu (corrected)
        }
        // Force host→device sync for GPU builds
        material_params_.Read();
    }

    ~SEMVisco_IsotropicElasticIntegrator3D() override = default;

    // Set members from base class and compute DOF ordering for vector field
    void AssemblePA(const FiniteElementSpace &fes) override {
        SEMIntegratorBase3D::AssemblePA(fes);
        dofs_.ComputeVector(fes, order_);
    }

    //For the reason of vectorization, using the macro dispatch here,
    //huge switch codes are generated for different orders....
    //This is the definition and declaration to call the AddMultPA_Visco_Opt kernel declared below.
    void AddMultPA(const Vector &x, Vector &y) const override {
        SEM_DISPATCH_NGLL_3D(ngll_, AddMultPA_Visco_Opt, x, y);
    }

    /// Get attenuation model
    GeneralizedZener3D& GetAttenuation() { return attenuation_; }
    const GeneralizedZener3D& GetAttenuation() const { return attenuation_; }

    /// Check if attenuation is enabled
    bool HasAttenuation() const { return attenuation_.HasMemory(); }

    // =========================================================================
    // Attenuation interface (override from base class)
    // =========================================================================

    bool HasAttenuationModel() const override { return HasAttenuation(); }

    void ResetAttenuationState() override { attenuation_.Reset(); }

    const Vector& GetMaterialParams() const { return material_params_; }

    /// material_params_ is stored in comp->ix->iy->iz->e order for GPU coalesced access
    /// This view is just a convenient wrapper to access the data using (comp, ix, iy, iz, e) indexing,
    /// otherwise need to use like ic + 2*(ix + ngll*(iy + ngll*(iz + ngll*e))) to access the data in the kernel
    /// Component 0 = kappa (bulk modulus, corrected), Component 1 = mu (shear modulus, corrected)
    auto ViewMaterialParams() const {
        return Reshape(material_params_.Read(), 2, ngll_, ngll_, ngll_, ne_);
    }

    /// Get kappa/mu vectors (for passing to EnableAttenuation)
    Vector& KappaVec() { return kappa_vec_; }
    Vector& MuVec() { return mu_vec_; }

    size_t MemoryUsage() const override {
        return SEMIntegratorBase3D::MemoryUsage() +
               material_params_.Size() * sizeof(real_t) +
               attenuation_.MemoryUsage();
    }

    // Kernel methods (public for CUDA extended lambda)
    // This will be called by AddMultPA with macro dispatch for different NGLL values, and the actual implementation will be in the .cpp file.
    template<int NGLL>
    void AddMultPA_Visco_Opt(const Vector &x, Vector &y) const;

protected:
    Vector material_params_;
    GeneralizedZener3D attenuation_;
    Vector kappa_vec_;  ///< Kappa vector (modified by EnableAttenuation)
    Vector mu_vec_;     ///< Mu vector (modified by EnableAttenuation)
};


}  // namespace SEM

#endif  // SEM_VISCO_ISOTROPIC_ELASTIC_INTEGRATOR_HPP
