/**
 * @file AcousticMaterialBase.hpp
 * @brief Base interfaces for acoustic material types (2D and 3D)
 *
 * Contains:
 * - AcousticMaterialBase2D: Interface for 2D acoustic materials
 * - AcousticMaterialBase3D: Interface for 3D acoustic materials
 *
 * Acoustic materials provide:
 * - Kappa (bulk modulus)
 * - InvRho (inverse density, 1/rho)
 * - Optional: Qkappa (attenuation Q factor for bulk modulus)
 *
 * Note: Acoustic materials have no Qmu since there are no shear waves.
 */

#ifndef SEM_ACOUSTIC_MATERIAL_BASE_HPP
#define SEM_ACOUSTIC_MATERIAL_BASE_HPP

#include "material/MaterialBase.hpp"
#include "material/MaterialField.hpp"
#include <mfem.hpp>

namespace SEM {

using namespace mfem;

// =============================================================================
// AcousticMaterialBase2D
// =============================================================================

/**
 * @class AcousticMaterialBase2D
 * @brief Abstract interface for 2D acoustic materials
 *
 * Concrete implementations:
 * - IsotropicAcousticMaterial
 */
class AcousticMaterialBase2D : public MaterialBase {
public:
    virtual ~AcousticMaterialBase2D() = default;

    // =========================================================================
    // Required acoustic properties (pure virtual)
    // =========================================================================

    /// Bulk modulus (kappa = rho * vp^2)
    virtual const MaterialField& Kappa() const = 0;

    /// Inverse density (1/rho)
    virtual const MaterialField& InvRho() const = 0;

    // =========================================================================
    // Mutable access for setup
    // =========================================================================

    virtual MaterialField& Kappa() = 0;
    virtual MaterialField& InvRho() = 0;

    // =========================================================================
    // Optional attenuation (Q factor for bulk modulus only)
    // =========================================================================

    /// Q factor for bulk modulus (requires HasAttenuation() == true)
    virtual const MaterialField& Qkappa() const {
        MFEM_ABORT("Qkappa requires attenuation enabled");
        return Kappa();  // unreachable
    }

    /// Mutable Q factor for bulk modulus (requires HasAttenuation() == true)
    virtual MaterialField& Qkappa() {
        MFEM_ABORT("Qkappa requires attenuation enabled");
        return Kappa();  // unreachable
    }

    /// Reference frequency for Q fitting (Hz)
    virtual real_t AttenuationF0() const { return 0.0; }

    /// Number of SLS relaxation mechanisms
    virtual int AttenuationNumUnits() const { return 0; }

    // NOTE: ApplyAttenuationCorrection() is declared on MaterialBase.
    // Acoustic materials override it to apply unrelaxed correction to kappa
    // using SLS parameters (physical dispersion correction).


    // =========================================================================
    // Dimension info (stored in base class for efficiency)
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
    /// Acoustic: Vp = sqrt(kappa / rho) = sqrt(kappa * inv_rho)
    real_t GetMaxVelocity() const override = 0;

    /// Minimum P-wave velocity for a specific element (for per-element PPW check)
    /// Returns min(Vp) among all GLL points within the element.
    real_t GetElementMinVelocity(int e) const override = 0;

    /// Maximum P-wave velocity for a specific element (for per-element CFL check)
    /// Returns max(Vp) among all GLL points within the element.
    real_t GetElementMaxVelocity(int e) const override = 0;

protected:
    AcousticMaterialBase2D(int ne, int ngllx, int nglly)
        : ne_(ne), ngllx_(ngllx), nglly_(nglly) {}

    int ne_;
    int ngllx_;
    int nglly_;
};


// =============================================================================
// AcousticMaterialBase3D
// =============================================================================

/**
 * @class AcousticMaterialBase3D
 * @brief Abstract interface for 3D acoustic materials
 */
class AcousticMaterialBase3D : public MaterialBase {
public:
    virtual ~AcousticMaterialBase3D() = default;

    // =========================================================================
    // Required acoustic properties (pure virtual)
    // =========================================================================

    /// Bulk modulus (kappa = rho * vp^2)
    virtual const MaterialField3D& Kappa() const = 0;

    /// Inverse density (1/rho)
    virtual const MaterialField3D& InvRho() const = 0;

    // =========================================================================
    // Mutable access for setup
    // =========================================================================

    virtual MaterialField3D& Kappa() = 0;
    virtual MaterialField3D& InvRho() = 0;

    // =========================================================================
    // Optional attenuation (Q factor for bulk modulus only)
    // =========================================================================

    /// Q factor for bulk modulus (requires HasAttenuation() == true)
    virtual const MaterialField3D& Qkappa() const {
        MFEM_ABORT("Qkappa requires attenuation enabled");
        return Kappa();  // unreachable
    }

    /// Mutable Q factor for bulk modulus (requires HasAttenuation() == true)
    virtual MaterialField3D& Qkappa() {
        MFEM_ABORT("Qkappa requires attenuation enabled");
        return Kappa();  // unreachable
    }

    /// Reference frequency for Q fitting (Hz)
    virtual real_t AttenuationF0() const { return 0.0; }

    /// Number of SLS relaxation mechanisms
    virtual int AttenuationNumUnits() const { return 0; }

    // NOTE: ApplyAttenuationCorrection() is declared on MaterialBase.
    // Acoustic materials override it to apply unrelaxed correction to kappa
    // using SLS parameters (physical dispersion correction).


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
    /// Acoustic: Vp = sqrt(kappa / rho) = sqrt(kappa * inv_rho)
    real_t GetMaxVelocity() const override = 0;

    /// Minimum P-wave velocity for a specific element (for per-element PPW check)
    /// Returns min(Vp) among all GLL points within the element.
    real_t GetElementMinVelocity(int e) const override = 0;

    /// Maximum P-wave velocity for a specific element (for per-element CFL check)
    /// Returns max(Vp) among all GLL points within the element.
    real_t GetElementMaxVelocity(int e) const override = 0;

protected:
    AcousticMaterialBase3D(int ne, int ngllx, int nglly, int ngllz)
        : ne_(ne), ngllx_(ngllx), nglly_(nglly), ngllz_(ngllz) {}

    int ne_;
    int ngllx_;
    int nglly_;
    int ngllz_;
};

}  // namespace SEM

#endif  // SEM_ACOUSTIC_MATERIAL_BASE_HPP
