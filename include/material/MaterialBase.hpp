/**
 * @file MaterialBase.hpp
 * @brief Abstract base class for all material types
 *
 * Provides common interface:
 * - GetType(): Material type identifier
 * - MemoryUsage(): Memory footprint in bytes
 * - HasAttenuation(): Whether Q factors are available
 * - GetDomainType(): Solid or Fluid domain
 */

#ifndef SEM_MATERIAL_BASE_HPP
#define SEM_MATERIAL_BASE_HPP

#include "common/Types.hpp"
#include <mfem.hpp>

namespace SEM {

using mfem::real_t;

/**
 * @class MaterialBase
 * @brief Abstract base for all material models
 *
 * This is the root of the material hierarchy. Specific interfaces
 * (ElasticMaterialBase, AcousticMaterialBase) inherit from this.
 */
class MaterialBase {
public:
    virtual ~MaterialBase() = default;

    // No copy (materials are typically large)
    MaterialBase(const MaterialBase&) = delete;
    MaterialBase& operator=(const MaterialBase&) = delete;

    // Move is OK
    MaterialBase(MaterialBase&&) = default;
    MaterialBase& operator=(MaterialBase&&) = default;

    /// Material type identifier
    virtual MaterialType GetType() const = 0;

    /// Memory usage in bytes
    virtual size_t MemoryUsage() const = 0;

    /// Whether attenuation (Q factors) is enabled
    virtual bool HasAttenuation() const { return false; }

    /// Attenuation reference frequency (Hz)
    virtual real_t AttenuationF0() const { return 0.0; }

    /// Number of standard linear solid units for attenuation
    virtual int AttenuationNumUnits() const { return 0; }

    /// Representative Q factor for bulk modulus (for diagnostics)
    virtual real_t AttenuationQkappa() const { return 0.0; }

    /// Representative Q factor for shear modulus (for diagnostics, 0 for acoustic)
    virtual real_t AttenuationQmu() const { return 0.0; }

    /// Domain type derived from material type
    DomainType GetDomainType() const {
        return GetDomainFromMaterial(GetType());
    }

    /// Number of elements
    virtual int NumElements() const = 0;

    /// Maximum wave velocity across all elements (for CFL calculation)
    /// For elastic: max(Vp), for acoustic: max(Vp)
    virtual real_t GetMaxVelocity() const = 0;

    /// Minimum wave velocity across all elements (for wavelength sampling check)
    /// Note: Physical meaning differs by material type:
    /// - Elastic: min(Vs) - the slowest wave for spatial resolution
    /// - Acoustic: min(Vp) - the only wave type present
    virtual real_t GetMinVelocity() const = 0;

    /// Minimum wave velocity for a specific element (for per-element PPW check)
    /// Returns the minimum velocity among all GLL points within the element.
    /// - Elastic: min(Vs) within element e
    /// - Acoustic: min(Vp) within element e
    virtual real_t GetElementMinVelocity(int e) const = 0;

    /// Maximum wave velocity for a specific element (for per-element CFL check)
    /// Returns the maximum velocity among all GLL points within the element.
    /// - Elastic: max(Vp) within element e
    /// - Acoustic: max(Vp) within element e
    virtual real_t GetElementMaxVelocity(int e) const = 0;

    /// Apply attenuation correction to moduli (convert to unrelaxed values).
    /// No-op when attenuation is disabled. Concrete materials override this
    /// to compute unrelaxed kappa (acoustic) or unrelaxed kappa/mu (elastic)
    /// using their stored Q fields and SLS parameters.
    virtual void ApplyAttenuationCorrection() {}

    // Note: stiffness integrator construction is handled by the free-function
    // factories in integ/StiffnessIntegratorFactory.hpp. MaterialBase does not
    // declare a virtual for this because the operator layer already carries
    // the physics/dimension type information needed to dispatch directly.

protected:
    MaterialBase() = default;
};

}  // namespace SEM

#endif  // SEM_MATERIAL_BASE_HPP
