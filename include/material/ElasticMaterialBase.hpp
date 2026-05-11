/**
 * @file ElasticMaterialBase.hpp
 * @brief Base interfaces for elastic material types (2D and 3D)
 *
 * Contains:
 * - ElasticMaterialBase2D: Interface for 2D elastic materials
 * - ElasticMaterialBase3D: Interface for 3D elastic materials
 *
 * Elastic materials provide:
 * - Kappa (bulk modulus)
 * - Mu (shear modulus)
 * - Rho (density)
 * - Optional: Qkappa, Qmu (attenuation Q factors)
 *
 * Note: Lambda (Lamé's first parameter) is isotropic-specific and defined
 * in IsotropicElasticMaterial, not in the base class.
 */

#ifndef SEM_ELASTIC_MATERIAL_BASE_HPP
#define SEM_ELASTIC_MATERIAL_BASE_HPP

#include "material/MaterialBase.hpp"
#include "material/MaterialField.hpp"
#include <mfem.hpp>

namespace SEM {

using namespace mfem;

// =============================================================================
// ElasticMaterialBase2D
// =============================================================================

/**
 * @class ElasticMaterialBase2D
 * @brief Abstract interface for 2D elastic materials
 *
 * Concrete implementations:
 * - IsotropicElasticMaterial
 * - (Future: AnisotropicElasticMaterial)
 */
class ElasticMaterialBase2D : public MaterialBase {
public:
    virtual ~ElasticMaterialBase2D() = default;

    // =========================================================================
    // Required elastic properties (pure virtual)
    // =========================================================================

    /// Bulk modulus (kappa)
    virtual const MaterialField& Kappa() const = 0;

    /// Shear modulus
    virtual const MaterialField& Mu() const = 0;

    /// Density
    virtual const MaterialField& Rho() const = 0;

    // =========================================================================
    // Mutable access for setup
    // =========================================================================

    virtual MaterialField& Kappa() = 0;
    virtual MaterialField& Mu() = 0;
    virtual MaterialField& Rho() = 0;

    // =========================================================================
    // Optional attenuation (Q factors)
    // =========================================================================

    /// Q factor for bulk modulus (requires HasAttenuation() == true)
    virtual const MaterialField& Qkappa() const {
        MFEM_ABORT("Qkappa requires attenuation enabled");
        return Mu();  // unreachable
    }

    /// Q factor for shear modulus (requires HasAttenuation() == true)
    virtual const MaterialField& Qmu() const {
        MFEM_ABORT("Qmu requires attenuation enabled");
        return Mu();  // unreachable
    }

    /// Mutable Q factor for bulk modulus (requires HasAttenuation() == true)
    virtual MaterialField& Qkappa() {
        MFEM_ABORT("Qkappa requires attenuation enabled");
        return Mu();  // unreachable
    }

    /// Mutable Q factor for shear modulus (requires HasAttenuation() == true)
    virtual MaterialField& Qmu() {
        MFEM_ABORT("Qmu requires attenuation enabled");
        return Mu();  // unreachable
    }

    /// Reference frequency for Q fitting (Hz)
    virtual real_t AttenuationF0() const { return 0.0; }

    /// Number of SLS relaxation mechanisms
    virtual int AttenuationNumUnits() const { return 0; }

    // NOTE: ApplyAttenuationCorrection() is declared on MaterialBase.
    // Elastic materials override it to apply unrelaxed correction to kappa
    // and mu using SLS parameters (shift_velocities_from_f0).


    // =========================================================================
    // Dimension info
    // =========================================================================

    int NumGLLx() const { return ngllx_; }
    int NumGLLy() const { return nglly_; }
    int NumElements() const override { return ne_; }

    /// Print memory usage and statistics
    virtual void PrintInfo(std::ostream& os = mfem::out) const;

    // =========================================================================
    // Maximum velocity (pure virtual - implemented by derived classes)
    // =========================================================================

    /// Maximum P-wave velocity across all elements (for CFL calculation)
    /// Implementation depends on material type (isotropic, anisotropic, etc.)
    real_t GetMaxVelocity() const override = 0;

    /// Minimum S-wave velocity for a specific element (for per-element PPW check)
    /// Returns min(Vs) among all GLL points within the element.
    real_t GetElementMinVelocity(int e) const override = 0;

    /// Maximum P-wave velocity for a specific element (for per-element CFL check)
    /// Returns max(Vp) among all GLL points within the element.
    real_t GetElementMaxVelocity(int e) const override = 0;

protected:
    ElasticMaterialBase2D(int ne, int ngllx, int nglly)
        : ne_(ne), ngllx_(ngllx), nglly_(nglly) {}

    int ne_;
    int ngllx_;
    int nglly_;
};


// =============================================================================
// ElasticMaterialBase3D
// =============================================================================

/**
 * @class ElasticMaterialBase3D
 * @brief Abstract interface for 3D elastic materials
 */
class ElasticMaterialBase3D : public MaterialBase {
public:
    virtual ~ElasticMaterialBase3D() = default;

    // =========================================================================
    // Required elastic properties (pure virtual)
    // =========================================================================

    /// Bulk modulus (kappa)
    virtual const MaterialField3D& Kappa() const = 0;

    /// Shear modulus
    virtual const MaterialField3D& Mu() const = 0;

    /// Density
    virtual const MaterialField3D& Rho() const = 0;

    // =========================================================================
    // Mutable access for setup
    // =========================================================================

    virtual MaterialField3D& Kappa() = 0;
    virtual MaterialField3D& Mu() = 0;
    virtual MaterialField3D& Rho() = 0;

    // =========================================================================
    // Optional attenuation (Q factors)
    // =========================================================================

    /// Q factor for bulk modulus (requires HasAttenuation() == true)
    virtual const MaterialField3D& Qkappa() const {
        MFEM_ABORT("Qkappa requires attenuation enabled");
        return Mu();  // unreachable
    }

    /// Q factor for shear modulus (requires HasAttenuation() == true)
    virtual const MaterialField3D& Qmu() const {
        MFEM_ABORT("Qmu requires attenuation enabled");
        return Mu();  // unreachable
    }

    /// Mutable Q factor for bulk modulus (requires HasAttenuation() == true)
    virtual MaterialField3D& Qkappa() {
        MFEM_ABORT("Qkappa requires attenuation enabled");
        return Mu();  // unreachable
    }

    /// Mutable Q factor for shear modulus (requires HasAttenuation() == true)
    virtual MaterialField3D& Qmu() {
        MFEM_ABORT("Qmu requires attenuation enabled");
        return Mu();  // unreachable
    }

    /// Reference frequency for Q fitting (Hz)
    virtual real_t AttenuationF0() const { return 0.0; }

    /// Number of SLS relaxation mechanisms
    virtual int AttenuationNumUnits() const { return 0; }

    // NOTE: ApplyAttenuationCorrection() is declared on MaterialBase.
    // Elastic materials override it to apply unrelaxed correction to kappa
    // and mu using SLS parameters (shift_velocities_from_f0).


    // =========================================================================
    // Dimension info (stored in base class for efficiency)
    // =========================================================================

    int NumGLLx() const { return ngllx_; }
    int NumGLLy() const { return nglly_; }
    int NumGLLz() const { return ngllz_; }
    int NumElements() const override { return ne_; }

    /// Print memory usage and statistics
    virtual void PrintInfo(std::ostream& os = mfem::out) const;

    // =========================================================================
    // Maximum velocity (pure virtual - implemented by derived classes)
    // =========================================================================

    /// Maximum P-wave velocity across all elements (for CFL calculation)
    /// Implementation depends on material type (isotropic, anisotropic, etc.)
    real_t GetMaxVelocity() const override = 0;

    /// Minimum S-wave velocity for a specific element (for per-element PPW check)
    /// Returns min(Vs) among all GLL points within the element.
    real_t GetElementMinVelocity(int e) const override = 0;

    /// Maximum P-wave velocity for a specific element (for per-element CFL check)
    /// Returns max(Vp) among all GLL points within the element.
    real_t GetElementMaxVelocity(int e) const override = 0;

protected:
    ElasticMaterialBase3D(int ne, int ngllx, int nglly, int ngllz)
        : ne_(ne), ngllx_(ngllx), nglly_(nglly), ngllz_(ngllz) {}

    int ne_;
    int ngllx_;
    int nglly_;
    int ngllz_;
};

}  // namespace SEM

#endif  // SEM_ELASTIC_MATERIAL_BASE_HPP
