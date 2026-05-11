/**
 * @file IsotropicElasticMaterial.hpp
 * @brief Isotropic elastic material classes (2D and 3D) with optional attenuation
 *
 * Contains:
 * - IsotropicElasticMaterial (2D): Lambda, mu, rho + optional Qkappa, Qmu
 * - IsotropicElasticMaterial3D: Lambda, mu, rho + optional Qkappa, Qmu
 *
 * When attenuation is enabled (HasAttenuation() == true), the material also
 * stores Qkappa, Qmu fields and attenuation parameters (f0, n_units).
 */

#ifndef SEM_ISOTROPIC_ELASTIC_MATERIAL_HPP
#define SEM_ISOTROPIC_ELASTIC_MATERIAL_HPP

#include "material/ElasticMaterialBase.hpp"
#include <memory>

namespace SEM {

// Forward declaration
struct MaterialConfig;

// =============================================================================
// 2D Isotropic Elastic Material
// =============================================================================

/**
 * @class IsotropicElasticMaterial
 * @brief 2D isotropic elastic material with optional attenuation
 *
 * Core properties: kappa (bulk modulus), lambda, mu, rho
 * Kappa is computed from lambda and mu: kappa = lambda + mu (2D plane strain)
 * Optional attenuation: Qkappa, Qmu, f0, n_units (when HasAttenuation() == true)
 */
class IsotropicElasticMaterial : public ElasticMaterialBase2D {
public:
    /// Construct with dimensions only (fields uninitialized)
    IsotropicElasticMaterial(int ne, int ngllx, int nglly);

    /// Construct and initialize from Vp, Vs, Rho coefficients
    IsotropicElasticMaterial(Coefficient& vp, Coefficient& vs, Coefficient& rho,
                             const ParFiniteElementSpace& fes,
                             const IntegrationRule& ir);

    /// Construct and initialize from Lame parameters
    IsotropicElasticMaterial(Coefficient& lambda, Coefficient& mu, Coefficient& rho,
                             const ParFiniteElementSpace& fes,
                             const IntegrationRule& ir,
                             bool lame_params);  // tag to distinguish from Vp/Vs constructor

    ~IsotropicElasticMaterial() override = default;

    // =========================================================================
    // Factory method from configuration
    // =========================================================================

    /**
     * @brief Create material from configuration
     *
     * Handles all format types: constant, ascii, hdf5, by_attribute.
     * If config.attenuation.enabled is true, also initializes Qkappa/Qmu fields.
     *
     * @param config Material configuration (includes attenuation settings)
     * @param fes Finite element space (scalar)
     * @param ir Integration rule
     * @return Unique pointer to created material
     */
    static std::unique_ptr<IsotropicElasticMaterial> FromConfig(
        const MaterialConfig& config,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    // =========================================================================
    // MaterialBase interface
    // =========================================================================

    MaterialType GetType() const override { return MaterialType::IsotropicElastic; }

    // =========================================================================
    // Elastic property access
    // =========================================================================

    const MaterialField& Kappa() const override { return kappa_; }
    const MaterialField& Mu() const override { return mu_; }
    const MaterialField& Rho() const override { return rho_; }

    MaterialField& Kappa() override { return kappa_; }
    MaterialField& Mu() override { return mu_; }
    MaterialField& Rho() override { return rho_; }

    /// Lambda (Lamé's first parameter) - isotropic specific
    const MaterialField& Lambda() const { return lambda_; }
    MaterialField& Lambda() { return lambda_; }

    // =========================================================================
    // Attenuation property access
    // =========================================================================

    /// Check if attenuation is enabled
    bool HasAttenuation() const override { return qkappa_ != nullptr; }

    /// Q factor for bulk modulus (only valid if HasAttenuation())
    const MaterialField& Qkappa() const override {
        MFEM_VERIFY(qkappa_, "Qkappa not available - attenuation not enabled");
        return *qkappa_;
    }

    /// Q factor for shear modulus (only valid if HasAttenuation())
    const MaterialField& Qmu() const override {
        MFEM_VERIFY(qmu_, "Qmu not available - attenuation not enabled");
        return *qmu_;
    }

    /// Mutable Q factor for bulk modulus (only valid if HasAttenuation())
    MaterialField& Qkappa() override {
        MFEM_VERIFY(qkappa_, "Qkappa not available - attenuation not enabled");
        return *qkappa_;
    }

    /// Mutable Q factor for shear modulus (only valid if HasAttenuation())
    MaterialField& Qmu() override {
        MFEM_VERIFY(qmu_, "Qmu not available - attenuation not enabled");
        return *qmu_;
    }

    /// Reference frequency for Q fitting (Hz)
    real_t AttenuationF0() const override { return f0_; }

    /// Number of SLS units (N_SLS)
    int AttenuationNumUnits() const override { return n_units_; }

    /// Representative Qkappa (first GLL point value, for diagnostics)
    real_t AttenuationQkappa() const override {
        return qkappa_ ? qkappa_->Data()(0) : 0.0;
    }

    /// Representative Qmu (first GLL point value, for diagnostics)
    real_t AttenuationQmu() const override {
        return qmu_ ? qmu_->Data()(0) : 0.0;
    }

    /// Apply attenuation correction to kappa and mu (unrelaxed moduli)
    void ApplyAttenuationCorrection() override;

    /// Per-GLL c_κ = κ_u / κ_user (populated in ApplyAttenuationCorrection
    /// from Q_κ; nullptr when attenuation is disabled).
    const MaterialField* UnrelaxedCorrectionKappa() const {
        return unrelaxed_correction_kappa_.get();
    }

    /// Per-GLL c_μ = μ_u / μ_user (populated from Q_μ; nullptr otherwise).
    const MaterialField* UnrelaxedCorrectionMu() const {
        return unrelaxed_correction_mu_.get();
    }

    // =========================================================================
    // Memory usage
    // =========================================================================

    size_t MemoryUsage() const override {
        size_t mem = kappa_.MemoryUsage() + lambda_.MemoryUsage() + mu_.MemoryUsage() + rho_.MemoryUsage();
        if (qkappa_) mem += qkappa_->MemoryUsage();
        if (qmu_) mem += qmu_->MemoryUsage();
        if (unrelaxed_correction_kappa_) mem += unrelaxed_correction_kappa_->MemoryUsage();
        if (unrelaxed_correction_mu_)    mem += unrelaxed_correction_mu_->MemoryUsage();
        return mem;
    }

    // =========================================================================
    // Velocity accessors
    // =========================================================================

    /// Maximum P-wave velocity: max(sqrt((lambda + 2*mu) / rho))
    real_t GetMaxVelocity() const override;

    /// Minimum S-wave velocity: min(sqrt(mu / rho))
    real_t GetMinVelocity() const override;

    /// Minimum S-wave velocity for a specific element
    real_t GetElementMinVelocity(int e) const override;

    /// Maximum P-wave velocity for a specific element
    real_t GetElementMaxVelocity(int e) const override;

    // Stiffness integrator construction is performed by the free-function
    // CreateElasticStiffnessIntegrator{2,3}D() in integ/StiffnessIntegratorFactory.hpp.

    // =========================================================================
    // Initialization helpers
    // =========================================================================

    /// Initialize from Vp, Vs, Rho coefficients (computes lambda, mu, kappa internally)
    void InitializeFromVelocities(Coefficient& vp, Coefficient& vs, Coefficient& rho,
                                   const ParFiniteElementSpace& fes,
                                   const IntegrationRule& ir);

    /// Initialize from Lame parameters directly (also computes kappa)
    void InitializeFromLame(Coefficient& lambda_c, Coefficient& mu_c, Coefficient& rho_c,
                             const ParFiniteElementSpace& fes,
                             const IntegrationRule& ir);

    /// Initialize attenuation with constant Q values
    void InitializeAttenuationConstant(real_t qkappa_val, real_t qmu_val,
                                        real_t f0, int n_units);

    /// Initialize attenuation from coefficients (for spatially varying Q)
    void InitializeAttenuationFromCoefficient(Coefficient& qkappa_coef, Coefficient& qmu_coef,
                                               const ParFiniteElementSpace& fes,
                                               const IntegrationRule& ir,
                                               real_t f0, int n_units);

    /// Allocate attenuation fields only (for by_attribute where values are set per-element)
    void AllocateAttenuationFields(real_t f0, int n_units);

    /// Compute kappa from lambda and mu (call after lambda/mu are set)
    void ComputeKappaFromLambdaMu();


private:
    // Core elastic properties
    MaterialField kappa_;   ///< Bulk modulus: kappa = lambda + mu (2D)
    MaterialField lambda_;
    MaterialField mu_;
    MaterialField rho_;

    // Optional attenuation (nullptr if not enabled)
    std::unique_ptr<MaterialField> qkappa_;
    std::unique_ptr<MaterialField> qmu_;
    std::unique_ptr<MaterialField> unrelaxed_correction_kappa_;  ///< c_κ per GLL
    std::unique_ptr<MaterialField> unrelaxed_correction_mu_;     ///< c_μ per GLL
    real_t f0_ = 1.0;       ///< Reference frequency for Q fitting
    int n_units_ = 3;       ///< Number of SLS units
    bool moduli_corrected_ = false;  ///< True if kappa/mu have been corrected to unrelaxed
};


// =============================================================================
// 3D Isotropic Elastic Material
// =============================================================================

/**
 * @class IsotropicElasticMaterial3D
 * @brief 3D isotropic elastic material with optional attenuation
 *
 * Core properties: lambda, mu, rho
 * Optional attenuation: Qkappa, Qmu, f0, n_units (when HasAttenuation() == true)
 */
class IsotropicElasticMaterial3D : public ElasticMaterialBase3D {
public:
    IsotropicElasticMaterial3D(int ne, int ngllx, int nglly, int ngllz);

    IsotropicElasticMaterial3D(Coefficient& lambda_c, Coefficient& mu_c, Coefficient& rho_c,
                                const ParFiniteElementSpace& fes,
                                const IntegrationRule& ir);

    ~IsotropicElasticMaterial3D() override = default;

    // =========================================================================
    // Factory method from configuration
    // =========================================================================

    /**
     * @brief Create 3D elastic material from configuration
     *
     * If config.attenuation.enabled is true, also initializes Qkappa/Qmu fields.
     */
    static std::unique_ptr<IsotropicElasticMaterial3D> FromConfig(
        const MaterialConfig& config,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    // =========================================================================
    // MaterialBase interface
    // =========================================================================

    MaterialType GetType() const override { return MaterialType::IsotropicElastic; }

    // =========================================================================
    // Elastic property access
    // =========================================================================

    const MaterialField3D& Kappa() const override { return kappa_; }
    const MaterialField3D& Mu() const override { return mu_; }
    const MaterialField3D& Rho() const override { return rho_; }

    MaterialField3D& Kappa() override { return kappa_; }
    MaterialField3D& Mu() override { return mu_; }
    MaterialField3D& Rho() override { return rho_; }

    /// Lambda (Lamé's first parameter) - isotropic specific
    const MaterialField3D& Lambda() const { return lambda_; }
    MaterialField3D& Lambda() { return lambda_; }

    // =========================================================================
    // Attenuation property access
    // =========================================================================

    /// Check if attenuation is enabled
    bool HasAttenuation() const override { return qkappa_ != nullptr; }

    /// Q factor for bulk modulus (only valid if HasAttenuation())
    const MaterialField3D& Qkappa() const override {
        MFEM_VERIFY(qkappa_, "Qkappa not available - attenuation not enabled");
        return *qkappa_;
    }

    /// Q factor for shear modulus (only valid if HasAttenuation())
    const MaterialField3D& Qmu() const override {
        MFEM_VERIFY(qmu_, "Qmu not available - attenuation not enabled");
        return *qmu_;
    }

    /// Mutable Q factor for bulk modulus (only valid if HasAttenuation())
    MaterialField3D& Qkappa() override {
        MFEM_VERIFY(qkappa_, "Qkappa not available - attenuation not enabled");
        return *qkappa_;
    }

    /// Mutable Q factor for shear modulus (only valid if HasAttenuation())
    MaterialField3D& Qmu() override {
        MFEM_VERIFY(qmu_, "Qmu not available - attenuation not enabled");
        return *qmu_;
    }

    /// Reference frequency for Q fitting (Hz)
    real_t AttenuationF0() const override { return f0_; }

    /// Number of SLS units (N_SLS)
    int AttenuationNumUnits() const override { return n_units_; }

    /// Representative Qkappa (first GLL point value, for diagnostics)
    real_t AttenuationQkappa() const override {
        return qkappa_ ? qkappa_->Data()(0) : 0.0;
    }

    /// Representative Qmu (first GLL point value, for diagnostics)
    real_t AttenuationQmu() const override {
        return qmu_ ? qmu_->Data()(0) : 0.0;
    }

    /// Apply attenuation correction to kappa and mu (unrelaxed moduli)
    void ApplyAttenuationCorrection() override;

    /// Per-GLL c_κ = κ_u / κ_user (populated in ApplyAttenuationCorrection
    /// from Q_κ; nullptr when attenuation is disabled).
    const MaterialField3D* UnrelaxedCorrectionKappa() const {
        return unrelaxed_correction_kappa_.get();
    }

    /// Per-GLL c_μ = μ_u / μ_user (populated from Q_μ; nullptr otherwise).
    const MaterialField3D* UnrelaxedCorrectionMu() const {
        return unrelaxed_correction_mu_.get();
    }

    // =========================================================================
    // Memory usage
    // =========================================================================

    size_t MemoryUsage() const override {
        size_t mem = kappa_.MemoryUsage() + lambda_.MemoryUsage() + mu_.MemoryUsage() + rho_.MemoryUsage();
        if (qkappa_) mem += qkappa_->MemoryUsage();
        if (qmu_) mem += qmu_->MemoryUsage();
        if (unrelaxed_correction_kappa_) mem += unrelaxed_correction_kappa_->MemoryUsage();
        if (unrelaxed_correction_mu_)    mem += unrelaxed_correction_mu_->MemoryUsage();
        return mem;
    }

    // =========================================================================
    // Velocity accessors
    // =========================================================================

    /// Maximum P-wave velocity: max(sqrt((lambda + 2*mu) / rho))
    real_t GetMaxVelocity() const override;

    /// Minimum S-wave velocity: min(sqrt(mu / rho))
    real_t GetMinVelocity() const override;

    /// Minimum S-wave velocity for a specific element
    real_t GetElementMinVelocity(int e) const override;

    /// Maximum P-wave velocity for a specific element
    real_t GetElementMaxVelocity(int e) const override;

    // Stiffness integrator construction is performed by the free-function
    // CreateElasticStiffnessIntegrator{2,3}D() in integ/StiffnessIntegratorFactory.hpp.

    // =========================================================================
    // Initialization helpers
    // =========================================================================

    /// Initialize attenuation with constant Q values
    void InitializeAttenuationConstant(real_t qkappa_val, real_t qmu_val,
                                        real_t f0, int n_units);

    /// Initialize attenuation from coefficients (for spatially varying Q)
    void InitializeAttenuationFromCoefficient(Coefficient& qkappa_coef, Coefficient& qmu_coef,
                                               const ParFiniteElementSpace& fes,
                                               const IntegrationRule& ir,
                                               real_t f0, int n_units);

    /// Allocate attenuation fields only (for by_attribute where values are set per-element)
    void AllocateAttenuationFields(real_t f0, int n_units);

    /// Compute kappa from lambda and mu (call after lambda/mu are set)
    /// 3D: kappa = lambda + (2/3)*mu
    void ComputeKappaFromLambdaMu();


private:
    // Core elastic properties
    MaterialField3D kappa_;   ///< Bulk modulus: kappa = lambda + (2/3)*mu (3D)
    MaterialField3D lambda_;
    MaterialField3D mu_;
    MaterialField3D rho_;

    // Optional attenuation (nullptr if not enabled)
    std::unique_ptr<MaterialField3D> qkappa_;
    std::unique_ptr<MaterialField3D> qmu_;
    std::unique_ptr<MaterialField3D> unrelaxed_correction_kappa_;  ///< c_κ per GLL
    std::unique_ptr<MaterialField3D> unrelaxed_correction_mu_;     ///< c_μ per GLL
    real_t f0_ = 1.0;       ///< Reference frequency for Q fitting
    int n_units_ = 3;       ///< Number of SLS units
    bool moduli_corrected_ = false;  ///< True if kappa/mu have been corrected to unrelaxed
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ELASTIC_MATERIAL_HPP
