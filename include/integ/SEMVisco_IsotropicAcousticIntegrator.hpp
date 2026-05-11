/**
 * @file SEMVisco_IsotropicAcousticIntegrator.hpp
 * @brief Viscoacoustic (isotropic acoustic with attenuation) stiffness integrator for SEM
 *
 * Isotropic acoustic with Generalized Zener (SLS) attenuation.
 * Uses Newmark scheme for viscoacoustic attenuation.
 *
 * Wave equation (displacement potential):
 *   (1/κ_u) ∂²φ/∂t² = ∇·((1/ρ)∇φ) + Σᵢ eᵢ
 *
 * Memory variable ODE:
 *   ∂eᵢ/∂t = -eᵢ/τ_σ,i + φᵢ × ∇²φ
 *
 * Newmark scheme:
 *   eᵢ = A² × eᵢ + B × (dot_e1 + A × dot_e1_old)
 *   where dot_e1 = stiffness operator output
 */

#ifndef SEM_VISCO_ISOTROPIC_ACOUSTIC_INTEGRATOR_HPP
#define SEM_VISCO_ISOTROPIC_ACOUSTIC_INTEGRATOR_HPP

#include <mfem.hpp>
#include "general/forall.hpp"  // MFEM 4.8 mfem.hpp drops this; needed for mfem::Reshape
#include "integ/core/SEMIntegratorBase.hpp"
#include "integ/core/SEMKernelDispatch.hpp"
#include "integ/attenuation/GeneralizedZener.hpp"

namespace SEM {


using namespace mfem;

/**
 * @class SEMVisco_IsotropicAcousticIntegrator2D
 * @brief 2D Viscoacoustic stiffness integrator (Crank-Nicolson scheme)
 */
class SEMVisco_IsotropicAcousticIntegrator2D : public SEMIntegratorBase2D {
public:
    /**
     * @brief Constructor with inverse density and kappa vectors
     * @param inv_rho Inverse density (1/ρ) per GLL point
     * @param kappa Bulk modulus per GLL point (copied, will be modified with correction)
     *
     * The kappa vector is copied internally and will be modified by EnableAttenuation()
     * to apply the unrelaxed modulus correction.
     */
    SEMVisco_IsotropicAcousticIntegrator2D(const Vector& inv_rho, const Vector& kappa)
        : inv_rho_vec_(inv_rho), kappa_vec_(kappa)
    {
        int n = inv_rho.Size();
        material_params_.SetSize(2 * n);
        material_params_.UseDevice(true);
        inv_rho_vec_.UseDevice(true);
        kappa_vec_.UseDevice(true);
        // Don't copy yet - will be copied after EnableAttenuation applies correction
    }

    /// Copy corrected inv_rho/kappa to material_params_ (call after EnableAttenuation)
    void FinalizeMaterialParams() {
        int n = inv_rho_vec_.Size();
        real_t* data = material_params_.HostWrite();
        const real_t* inv_rho = inv_rho_vec_.HostRead();
        const real_t* k = kappa_vec_.HostRead();
        for (int i = 0; i < n; i++) {
            data[2*i + 0] = inv_rho[i];  // 1/rho
            data[2*i + 1] = k[i];        // kappa (corrected)
        }
        // Force host→device sync for GPU builds
        material_params_.Read();
    }

    ~SEMVisco_IsotropicAcousticIntegrator2D() override = default;

    // Set members from base class and compute DOF ordering for scalar field
    void AssemblePA(const FiniteElementSpace &fes) override {
        SEMIntegratorBase2D::AssemblePA(fes);
        dofs_.ComputeScalar(fes, order_);
    }

    //For the reason of vectorization, using the macro dispatch here,
    //huge switch codes are generated for different orders....
    //This is the definition and declaration to call the AddMultPA_Visco_Opt kernel declared below.
    void AddMultPA(const Vector &x, Vector &y) const override {
        SEM_DISPATCH_NGLL(ngll_, AddMultPA_Visco_Opt, x, y);
    }

    /// Get attenuation model
    GeneralizedZenerAcoustic2D& GetAttenuation() { return attenuation_; }
    const GeneralizedZenerAcoustic2D& GetAttenuation() const { return attenuation_; }

    /// Check if attenuation is enabled
    bool HasAttenuation() const { return attenuation_.HasAttenuation(); }

    // =========================================================================
    // Attenuation interface (override from base class)
    // =========================================================================

    bool HasAttenuationModel() const override { return HasAttenuation(); }

    void ResetAttenuationState() override { attenuation_.Reset(); }

    const Vector& GetMaterialParams() const { return material_params_; }

    /// material_params_ is stored in comp->ix->iy->e order for GPU coalesced access
    /// This view is just a convenient wrapper to access the data using (comp, ix, iy, e) indexing,
    /// otherwise need to use like ic + 2*(ix + ngll*(iy + ngll*e)) to access the data in the kernel
    /// Component 0 = inv_rho (1/ρ), Component 1 = kappa (bulk modulus, corrected)
    auto ViewMaterialParams() const {
        return Reshape(material_params_.Read(), 2, ngll_, ngll_, ne_);
    }

    /// Get kappa vector (for passing to EnableAttenuation)
    Vector& KappaVec() { return kappa_vec_; }

    size_t MemoryUsage() const override {
        return SEMIntegratorBase2D::MemoryUsage() +
               material_params_.Size() * sizeof(real_t) +
               inv_rho_vec_.Size() * sizeof(real_t) +
               kappa_vec_.Size() * sizeof(real_t) +
               attenuation_.MemoryUsage();
    }

    // Kernel methods (public for CUDA extended lambda)
    // This will be called by AddMultPA with macro dispatch for different NGLL values, and the actual implementation will be in the .cpp file.
    template<int NGLL>
    void AddMultPA_Visco_Opt(const Vector &x, Vector &y) const;

protected:
    Vector material_params_;           ///< Packed [inv_rho, kappa] per GLL point
    Vector inv_rho_vec_;               ///< Inverse density (1/ρ)
    Vector kappa_vec_;                 ///< Bulk modulus (modified by EnableAttenuation)
    GeneralizedZenerAcoustic2D attenuation_;
};









/**
 * @class SEMVisco_IsotropicAcousticIntegrator3D
 * @brief 3D Viscoacoustic stiffness integrator (Crank-Nicolson scheme)
 */
class SEMVisco_IsotropicAcousticIntegrator3D : public SEMIntegratorBase3D {
public:
    /**
     * @brief Constructor with inverse density and kappa vectors
     * @param inv_rho Inverse density (1/ρ) per GLL point
     * @param kappa Bulk modulus per GLL point (copied, will be modified with correction)
     *
     * The kappa vector is copied internally and will be modified by EnableAttenuation()
     * to apply the unrelaxed modulus correction.
     */
    SEMVisco_IsotropicAcousticIntegrator3D(const Vector& inv_rho, const Vector& kappa)
        : inv_rho_vec_(inv_rho), kappa_vec_(kappa)
    {
        int n = inv_rho.Size();
        material_params_.SetSize(2 * n);
        material_params_.UseDevice(true);
        inv_rho_vec_.UseDevice(true);
        kappa_vec_.UseDevice(true);
        // Don't copy yet - will be copied after EnableAttenuation applies correction
    }

    /// Copy corrected inv_rho/kappa to material_params_ (call after EnableAttenuation)
    void FinalizeMaterialParams() {
        int n = inv_rho_vec_.Size();
        real_t* data = material_params_.HostWrite();
        const real_t* inv_rho = inv_rho_vec_.HostRead();
        const real_t* k = kappa_vec_.HostRead();
        for (int i = 0; i < n; i++) {
            data[2*i + 0] = inv_rho[i];  // 1/rho
            data[2*i + 1] = k[i];        // kappa (corrected)
        }
        // Force host→device sync for GPU builds
        material_params_.Read();
    }

    ~SEMVisco_IsotropicAcousticIntegrator3D() override = default;

    //Set members from base class and compute DOF ordering for scalar field  
    void AssemblePA(const FiniteElementSpace &fes) override {
        SEMIntegratorBase3D::AssemblePA(fes);
        dofs_.ComputeScalar(fes, order_);
    }

    //For the reason of vectorization, using the macro dispatch here,
    //huge switch codes are generated for different orders....
    //This is the definition and declaration to call the AddMultPA_Visco_Opt kernel declared below.
    void AddMultPA(const Vector &x, Vector &y) const override {
        SEM_DISPATCH_NGLL_3D(ngll_, AddMultPA_Visco_Opt, x, y);
    }

    /// Get attenuation model
    GeneralizedZenerAcoustic3D& GetAttenuation() { return attenuation_; }
    const GeneralizedZenerAcoustic3D& GetAttenuation() const { return attenuation_; }

    /// Check if attenuation is enabled
    bool HasAttenuation() const { return attenuation_.HasAttenuation(); }

    // =========================================================================
    // Attenuation interface (override from base class)
    // =========================================================================

    bool HasAttenuationModel() const override { return HasAttenuation(); }

    void ResetAttenuationState() override { attenuation_.Reset(); }

    const Vector& GetMaterialParams() const { return material_params_; }

    /// View for material params [2, ngll, ngll, ngll, ne] - access: params(comp, ix, iy, iz, e)
    /// Component 0 = inv_rho (1/ρ), Component 1 = kappa (bulk modulus, corrected)
    /// material_params_ is stored in comp->ix->iy->iz->e order for GPU coalesced access
    /// This view is just a convenient wrapper to access the data using (comp, ix, iy, iz, e) indexing, 
    /// otherwise need to use like ic + 2*(ix + ngll*(iy + ngll*(iz + ngll*e))) to access the data in the kernel
    auto ViewMaterialParams() const {
        return Reshape(material_params_.Read(), 2, ngll_, ngll_, ngll_, ne_);
    }

    /// Get kappa vector (for passing to EnableAttenuation)
    Vector& KappaVec() { return kappa_vec_; }

    size_t MemoryUsage() const override {
        return SEMIntegratorBase3D::MemoryUsage() +
               material_params_.Size() * sizeof(real_t) +
               inv_rho_vec_.Size() * sizeof(real_t) +
               kappa_vec_.Size() * sizeof(real_t) +
               attenuation_.MemoryUsage();
    }

    // Kernel methods (public for CUDA extended lambda)
    // This will be called by AddMultPA with macro dispatch for different NGLL values, and the actual implementation will be in the .cpp file.
    template<int NGLL>
    void AddMultPA_Visco_Opt(const Vector &x, Vector &y) const;

protected:
    Vector material_params_;           ///< Packed [inv_rho, kappa] per GLL point
    Vector inv_rho_vec_;               ///< Inverse density (1/ρ)
    Vector kappa_vec_;                 ///< Bulk modulus (modified by EnableAttenuation)
    GeneralizedZenerAcoustic3D attenuation_;
};


}  // namespace SEM

#endif  // SEM_VISCO_ISOTROPIC_ACOUSTIC_INTEGRATOR_HPP
