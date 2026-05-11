/**
 * @file IsotropicAcousticMaterial.hpp
 * @brief Isotropic acoustic material classes (2D and 3D) with optional attenuation
 *
 * Contains:
 * - IsotropicAcousticMaterial (2D): Kappa, inv_rho + optional Qkappa
 * - IsotropicAcousticMaterial3D: Kappa, inv_rho + optional Qkappa
 *
 * When attenuation is enabled (HasAttenuation() == true), the material also
 * stores Qkappa field and attenuation parameters (f0, n_units).
 * Note: Acoustic materials only have Qkappa (no Qmu since there are no shear waves).
 */

#ifndef SEM_ISOTROPIC_ACOUSTIC_MATERIAL_HPP
#define SEM_ISOTROPIC_ACOUSTIC_MATERIAL_HPP

#include "material/AcousticMaterialBase.hpp"
#include <memory>

namespace SEM {

// Forward declaration
struct MaterialConfig;

// =============================================================================
// 2D Isotropic Acoustic Material
// =============================================================================

/**
 * @class IsotropicAcousticMaterial
 * @brief 2D isotropic acoustic material with optional attenuation
 *
 * Core properties: kappa (bulk modulus), inv_rho (inverse density)
 * Optional attenuation: Qkappa, f0, n_units (when HasAttenuation() == true)
 */
class IsotropicAcousticMaterial : public AcousticMaterialBase2D {
public:
    /// Construct with dimensions only
    IsotropicAcousticMaterial(int ne, int ngllx, int nglly);

    /// Construct from Vp and Rho coefficients
    IsotropicAcousticMaterial(Coefficient& vp, Coefficient& rho,
                     const ParFiniteElementSpace& fes,
                     const IntegrationRule& ir);

    /// Construct from kappa and inv_rho coefficients
    IsotropicAcousticMaterial(Coefficient& kappa, Coefficient& invrho,
                     const ParFiniteElementSpace& fes,
                     const IntegrationRule& ir,
                     bool kappa_invrho);  // tag

    ~IsotropicAcousticMaterial() override = default;

    // =========================================================================
    // Factory method from configuration
    // =========================================================================

    /**
     * @brief Create isotropic acoustic material from configuration
     *
     * If config.attenuation.enabled is true, also initializes Qkappa field.
     */
    static std::unique_ptr<IsotropicAcousticMaterial> FromConfig(
        const MaterialConfig& config,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    // =========================================================================
    // MaterialBase interface
    // =========================================================================

    MaterialType GetType() const override { return MaterialType::IsotropicAcoustic; }

    // =========================================================================
    // Core property access
    // =========================================================================

    const MaterialField& Kappa() const override { return kappa_; }
    const MaterialField& InvRho() const override { return inv_rho_; }

    MaterialField& Kappa() override { return kappa_; }
    MaterialField& InvRho() override { return inv_rho_; }

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

    /// Mutable Q factor for bulk modulus (only valid if HasAttenuation())
    MaterialField& Qkappa() override {
        MFEM_VERIFY(qkappa_, "Qkappa not available - attenuation not enabled");
        return *qkappa_;
    }

    /// Reference frequency for Q fitting (Hz)
    real_t AttenuationF0() const override { return f0_; }

    /// Number of SLS units (N_SLS)
    int AttenuationNumUnits() const override { return n_units_; }

    /// Representative Qkappa (first GLL point value, for diagnostics)
    real_t AttenuationQkappa() const override {
        return qkappa_ ? qkappa_->Data()(0) : 0.0;
    }

    /// Apply attenuation correction to kappa (unrelaxed modulus)
    void ApplyAttenuationCorrection() override;


    // =========================================================================
    // Memory usage
    // =========================================================================

    size_t MemoryUsage() const override {
        size_t mem = kappa_.MemoryUsage() + inv_rho_.MemoryUsage();
        if (qkappa_) mem += qkappa_->MemoryUsage();
        if (unrelaxed_correction_) mem += unrelaxed_correction_->MemoryUsage();
        return mem;
    }

    // =========================================================================
    // Velocity accessors
    // =========================================================================

    /// Maximum P-wave velocity: max(sqrt(kappa * inv_rho))
    real_t GetMaxVelocity() const override;

    /// Minimum P-wave velocity: min(sqrt(kappa * inv_rho))
    /// Note: Acoustic only has P-waves, so min velocity = min(Vp)
    real_t GetMinVelocity() const override;

    /// Minimum P-wave velocity for a specific element
    real_t GetElementMinVelocity(int e) const override;

    /// Maximum P-wave velocity for a specific element
    real_t GetElementMaxVelocity(int e) const override;

    // Stiffness integrator construction is performed by the free-function
    // CreateAcousticStiffnessIntegrator{2,3}D() in integ/StiffnessIntegratorFactory.hpp.

    // =========================================================================
    // Initialization helpers
    // =========================================================================

    /// Initialize from Vp and Rho
    void InitializeFromVelocity(Coefficient& vp, Coefficient& rho,
                                const ParFiniteElementSpace& fes,
                                const IntegrationRule& ir);

    /// Initialize attenuation with constant Q value
    void InitializeAttenuationConstant(real_t qkappa_val, real_t f0, int n_units);

    /// Initialize attenuation from coefficient (for spatially varying Q)
    void InitializeAttenuationFromCoefficient(Coefficient& qkappa_coef,
                                               const ParFiniteElementSpace& fes,
                                               const IntegrationRule& ir,
                                               real_t f0, int n_units);

    /// Allocate attenuation field only (for by_attribute where values are set per-element)
    void AllocateAttenuationFields(real_t f0, int n_units);

    /// Per-GLL-point unrelaxed correction factor c_i = κ_u(i) / κ_user(i).
    /// Populated during ApplyAttenuationCorrection; returns nullptr when
    /// attenuation is not enabled (in which case κ_u == κ_user, c ≡ 1).
    const MaterialField* UnrelaxedCorrection() const {
        return unrelaxed_correction_.get();
    }

private:
    // Core acoustic properties
    MaterialField kappa_;
    MaterialField inv_rho_;

    // Optional attenuation (nullptr if not enabled)
    std::unique_ptr<MaterialField> qkappa_;
    std::unique_ptr<MaterialField> unrelaxed_correction_;  ///< c_i per GLL (κ_u / κ_user)
    real_t f0_ = 1.0;       ///< Reference frequency for Q fitting
    int n_units_ = 3;       ///< Number of SLS units
    bool kappa_corrected_ = false;  ///< True if kappa has been corrected to unrelaxed
};


// =============================================================================
// 3D Isotropic Acoustic Material
// =============================================================================

/**
 * @class IsotropicAcousticMaterial3D
 * @brief 3D isotropic acoustic material with optional attenuation
 *
 * Core properties: kappa (bulk modulus), inv_rho (inverse density)
 * Optional attenuation: Qkappa, f0, n_units (when HasAttenuation() == true)
 */
class IsotropicAcousticMaterial3D : public AcousticMaterialBase3D {
public:
    /// Construct with dimensions only
    IsotropicAcousticMaterial3D(int ne, int ngllx, int nglly, int ngllz);

    /// Construct from Vp and Rho coefficients
    IsotropicAcousticMaterial3D(Coefficient& vp, Coefficient& rho,
                       const ParFiniteElementSpace& fes,
                       const IntegrationRule& ir);

    ~IsotropicAcousticMaterial3D() override = default;

    // =========================================================================
    // Factory method from configuration
    // =========================================================================

    /**
     * @brief Create 3D isotropic acoustic material from configuration
     *
     * If config.attenuation.enabled is true, also initializes Qkappa field.
     */
    static std::unique_ptr<IsotropicAcousticMaterial3D> FromConfig(
        const MaterialConfig& config,
        ParFiniteElementSpace& fes,
        const IntegrationRule& ir);

    // =========================================================================
    // MaterialBase interface
    // =========================================================================

    MaterialType GetType() const override { return MaterialType::IsotropicAcoustic; }

    // =========================================================================
    // Core property access
    // =========================================================================

    const MaterialField3D& Kappa() const override { return kappa_; }
    const MaterialField3D& InvRho() const override { return inv_rho_; }

    MaterialField3D& Kappa() override { return kappa_; }
    MaterialField3D& InvRho() override { return inv_rho_; }

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

    /// Mutable Q factor for bulk modulus (only valid if HasAttenuation())
    MaterialField3D& Qkappa() override {
        MFEM_VERIFY(qkappa_, "Qkappa not available - attenuation not enabled");
        return *qkappa_;
    }

    /// Reference frequency for Q fitting (Hz)
    real_t AttenuationF0() const override { return f0_; }

    /// Number of SLS units (N_SLS)
    int AttenuationNumUnits() const override { return n_units_; }

    /// Representative Qkappa (first GLL point value, for diagnostics)
    real_t AttenuationQkappa() const override {
        return qkappa_ ? qkappa_->Data()(0) : 0.0;
    }

    /// Apply attenuation correction to kappa (unrelaxed modulus)
    void ApplyAttenuationCorrection() override;


    // =========================================================================
    // Memory usage
    // =========================================================================

    size_t MemoryUsage() const override {
        size_t mem = kappa_.MemoryUsage() + inv_rho_.MemoryUsage();
        if (qkappa_) mem += qkappa_->MemoryUsage();
        if (unrelaxed_correction_) mem += unrelaxed_correction_->MemoryUsage();
        return mem;
    }

    // =========================================================================
    // Velocity accessors
    // =========================================================================

    /// Maximum P-wave velocity: max(sqrt(kappa * inv_rho))
    real_t GetMaxVelocity() const override;

    /// Minimum P-wave velocity: min(sqrt(kappa * inv_rho))
    /// Note: Acoustic only has P-waves, so min velocity = min(Vp)
    real_t GetMinVelocity() const override;

    /// Minimum P-wave velocity for a specific element
    real_t GetElementMinVelocity(int e) const override;

    /// Maximum P-wave velocity for a specific element
    real_t GetElementMaxVelocity(int e) const override;

    // Stiffness integrator construction is performed by the free-function
    // CreateAcousticStiffnessIntegrator{2,3}D() in integ/StiffnessIntegratorFactory.hpp.

    // =========================================================================
    // Initialization helpers
    // =========================================================================

    /// Initialize from Vp and Rho
    void InitializeFromVelocity(Coefficient& vp, Coefficient& rho,
                                const ParFiniteElementSpace& fes,
                                const IntegrationRule& ir);

    /// Initialize attenuation with constant Q value
    void InitializeAttenuationConstant(real_t qkappa_val, real_t f0, int n_units);

    /// Initialize attenuation from coefficient (for spatially varying Q)
    void InitializeAttenuationFromCoefficient(Coefficient& qkappa_coef,
                                               const ParFiniteElementSpace& fes,
                                               const IntegrationRule& ir,
                                               real_t f0, int n_units);

    /// Allocate attenuation field only (for by_attribute where values are set per-element)
    void AllocateAttenuationFields(real_t f0, int n_units);

    /// Per-GLL-point unrelaxed correction factor (see 2D analog).
    const MaterialField3D* UnrelaxedCorrection() const {
        return unrelaxed_correction_.get();
    }

private:
    // Core acoustic properties
    MaterialField3D kappa_;
    MaterialField3D inv_rho_;

    // Optional attenuation (nullptr if not enabled)
    std::unique_ptr<MaterialField3D> qkappa_;
    std::unique_ptr<MaterialField3D> unrelaxed_correction_;  ///< c_i per GLL (κ_u / κ_user)
    real_t f0_ = 1.0;       ///< Reference frequency for Q fitting
    int n_units_ = 3;       ///< Number of SLS units
    bool kappa_corrected_ = false;  ///< True if kappa has been corrected to unrelaxed
};

}  // namespace SEM

#endif  // SEM_ISOTROPIC_ACOUSTIC_MATERIAL_HPP
