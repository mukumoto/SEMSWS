/**
 * @file AttenuationModel.hpp
 * @brief Base interface for attenuation/damping models
 *
 * Defines the interface for memory variable-based attenuation models.
 * Each attenuation model specifies:
 * - Number of memory variables per relaxation mechanism
 * - Memory variable update scheme
 * - Attenuation stress correction
 *
 * This enables viscoelastic, viscoacoustic, and other time-dependent
 * material behaviors via memory variable formulations.
 */

#ifndef SEM_ATTENUATION_MODEL_HPP
#define SEM_ATTENUATION_MODEL_HPP

#include <mfem.hpp>

namespace SEM {


using namespace mfem;

/**
 * @brief Compile-time attenuation model traits
 *
 * Each attenuation model type must specialize this template with:
 * - static constexpr int NUM_MEM_VARS: memory variables per mechanism
 * - static constexpr bool HAS_MEMORY: whether model has memory variables
 *
 * Example specialization:
 * @code
 * template<>
 * struct AttenuationTraits<GeneralizedZener2D> {
 *     static constexpr int NUM_MEM_VARS = 3;  // bulk + 2 deviatoric
 *     static constexpr bool HAS_MEMORY = true;
 * };
 * @endcode
 */
template<typename ModelType>
struct AttenuationTraits;


/**
 * @brief Forward declarations for attenuation models
 *
 * Full definitions are in the respective headers:
 * - NoAttenuation.hpp
 * - GeneralizedZener.hpp
 */

// Forward declarations (defined in NoAttenuation.hpp)
class NoAttenuation2D;
class NoAttenuation3D;

// Forward declarations (defined in GeneralizedZener.hpp)
class GeneralizedZener2D;
class GeneralizedZener3D;

/// Single relaxation for viscoacoustic (to be implemented)
class ViscoacousticSLS2D;
class ViscoacousticSLS3D;


// =============================================================================
// Trait Specializations
// =============================================================================

template<>
struct AttenuationTraits<NoAttenuation2D> {
    static constexpr int NUM_MEM_VARS = 0;
    static constexpr bool HAS_MEMORY = false;
    static constexpr int DIM = 2;
};

template<>
struct AttenuationTraits<NoAttenuation3D> {
    static constexpr int NUM_MEM_VARS = 0;
    static constexpr bool HAS_MEMORY = false;
    static constexpr int DIM = 3;
};

template<>
struct AttenuationTraits<GeneralizedZener2D> {
    // 2D viscoelastic: 3 memory variables per mechanism
    // M1: bulk (volumetric) memory for tr(ε)
    // M2: deviatoric d11 = εxx - εkk/2
    // M3: deviatoric d12 = εxy
    static constexpr int NUM_MEM_VARS = 3;
    static constexpr bool HAS_MEMORY = true;
    static constexpr int DIM = 2;
};

template<>
struct AttenuationTraits<GeneralizedZener3D> {
    // 3D viscoelastic: 6 memory variables per mechanism
    // M1: bulk (volumetric) memory for tr(ε)
    // M2: deviatoric d11 = εxx - εkk/3
    // M3: deviatoric d22 = εyy - εkk/3
    // M4: shear ε_xy
    // M5: shear ε_xz
    // M6: shear ε_yz
    static constexpr int NUM_MEM_VARS = 6;
    static constexpr bool HAS_MEMORY = true;
    static constexpr int DIM = 3;
};

template<>
struct AttenuationTraits<ViscoacousticSLS2D> {
    // Viscoacoustic: 1 memory variable (pressure memory)
    static constexpr int NUM_MEM_VARS = 1;
    static constexpr bool HAS_MEMORY = true;
    static constexpr int DIM = 2;
};

template<>
struct AttenuationTraits<ViscoacousticSLS3D> {
    static constexpr int NUM_MEM_VARS = 1;
    static constexpr bool HAS_MEMORY = true;
    static constexpr int DIM = 3;
};




/**
 * @brief 2D Viscoelastic memory state (packed arrays like 3D)
 *
 * Stores memory variables and relaxation coefficients for 2D viscoelastic
 * computation using the Generalized Zener model.
 *
 * Storage layout: component-first packed arrays for GPU coalescing
 *   M_packed: [3, ngllx, nglly, ne, n_units] - 3 components (bulk, d11, d12)
 *   coeffs_packed: [4, ngllx, nglly, ne, n_units] - 4 coefficients (Crank-Nicolson)
 *   strain_old_packed: [3, ngllx, nglly, ne] - 3 components
 */
class ViscoelasticMemory2D {
public:
    ViscoelasticMemory2D() = default;
    ViscoelasticMemory2D(int ne, int ngllx, int nglly, int n_units);

    void Reset();
    bool IsInitialized() const { return n_units_ > 0; }
    int NumUnits() const { return n_units_; }
    size_t MemoryUsage() const;

    /// Enable GPU memory for all vectors
    void EnableDevice() {
        M_packed.UseDevice(true);
        coeffs_packed.UseDevice(true);
        strain_old_packed.UseDevice(true);
    }

    /// Force host→device sync for all vectors
    void SyncToDevice() {
        M_packed.Read();
        coeffs_packed.Read();
        strain_old_packed.Read();
    }

    // =========================================================================
    // View Factory Methods (Zero-Overhead MFEM-style access)
    // =========================================================================

    /// Memory variable view [3, ngllx, nglly, ne, n_units]
    /// Components: 0=M1(bulk), 1=M2(d11), 2=M3(d12)
    auto ViewM() const { return Reshape(M_packed.Read(), 3, ngllx_, nglly_, ne_, n_units_); }
    auto ViewMWrite() { return Reshape(M_packed.ReadWrite(), 3, ngllx_, nglly_, ne_, n_units_); }

    /// Crank-Nicolson coefficient view [4, ngllx, nglly, ne, n_units]
    /// Components: 0=alpha_kappa, 1=alpha_mu, 2=strain_coeff_kappa, 3=strain_coeff_mu
    auto ViewCoeffs() const { return Reshape(coeffs_packed.Read(), 4, ngllx_, nglly_, ne_, n_units_); }

    /// Old strain view [3, ngllx, nglly, ne]
    /// Components: 0=ekk_old, 1=d11_old, 2=d12_old
    auto ViewStrainOld() const { return Reshape(strain_old_packed.Read(), 3, ngllx_, nglly_, ne_); }
    auto ViewStrainOldWrite() { return Reshape(strain_old_packed.ReadWrite(), 3, ngllx_, nglly_, ne_); }

    // =========================================================================
    // Packed Arrays
    // =========================================================================

    /// Memory variables: [3, ngllx, nglly, ne, n_units]
    /// Components: 0=M1(bulk), 1=M2(d11), 2=M3(d12)
    Vector M_packed;

    /// Crank-Nicolson coefficients: [4, ngllx, nglly, ne, n_units]
    /// Components: 0=alpha_kappa, 1=alpha_mu, 2=strain_coeff_kappa, 3=strain_coeff_mu
    Vector coeffs_packed;

    /// Old strain values: [3, ngllx, nglly, ne]
    /// Components: 0=ekk_old, 1=d11_old, 2=d12_old
    Vector strain_old_packed;

private:
    int ne_ = 0, ngllx_ = 0, nglly_ = 0, n_units_ = 0;
};


/**
 * @brief 3D Viscoelastic memory state
 *
 * Stores memory variables and relaxation coefficients for 3D viscoelastic
 * computation using the Generalized Zener model.
 *
 * Storage layout: x-first (ix varies fastest in memory)
 *   gll_idx = ix + ngllx*(iy + nglly*(iz + ngllz*e))
 *   For 5D arrays with n_units: unit + n_units * gll_idx
 */
class ViscoelasticMemory3D {
public:
    ViscoelasticMemory3D() = default;
    ViscoelasticMemory3D(int ne, int ngllx, int nglly, int ngllz, int n_units);

    void Reset();
    bool IsInitialized() const { return n_units_ > 0; }
    int NumUnits() const { return n_units_; }
    size_t MemoryUsage() const;

    /// Enable GPU memory for all vectors
    void EnableDevice() {
        M_packed.UseDevice(true);
        coeffs_packed.UseDevice(true);
        strain_old_packed.UseDevice(true);
    }

    /// Force host→device sync for all vectors (call after EnableDevice)
    void SyncToDevice() {
        coeffs_packed.Read();
        strain_old_packed.Read();
        M_packed.Read();
    }

    // =========================================================================
    // Packed Arrays for GPU-Coalesced Access
    // =========================================================================

    // Memory variables: [6, ngllx, nglly, ngllz, ne, n_units]
    // Components: 0=M1(bulk), 1=M2(d11), 2=M3(d22), 3=M4(d12), 4=M5(d13), 5=M6(d23)
    Vector M_packed;

    // Crank-Nicolson coefficients: [4, ngllx, nglly, ngllz, ne, n_units]
    // Components: 0=alpha_kappa, 1=alpha_mu, 2=strain_coeff_kappa, 3=strain_coeff_mu
    Vector coeffs_packed;

    // Old strain values: [6, ngllx, nglly, ngllz, ne]
    // Components: 0=ekk, 1=d11, 2=d22, 3=d12, 4=d13, 5=d23
    Vector strain_old_packed;

    // =========================================================================
    // View Factory Methods (Zero-Overhead MFEM-style access)
    // =========================================================================

    // Memory variable View [6, ngllx, nglly, ngllz, ne, n_units]
    // Access: view(comp, ix, iy, iz, e, unit) - comp fastest for GPU coalescing
    auto ViewM() const { return Reshape(M_packed.Read(), 6, ngllx_, nglly_, ngllz_, ne_, n_units_); }
    auto ViewMWrite() { return Reshape(M_packed.ReadWrite(), 6, ngllx_, nglly_, ngllz_, ne_, n_units_); }

    // Crank-Nicolson coefficient View [4, ngllx, nglly, ngllz, ne, n_units]
    // Components: 0=alpha_kappa, 1=alpha_mu, 2=strain_coeff_kappa, 3=strain_coeff_mu
    auto ViewCoeffs() const { return Reshape(coeffs_packed.Read(), 4, ngllx_, nglly_, ngllz_, ne_, n_units_); }

    // Old strain View [6, ngllx, nglly, ngllz, ne]
    // Components: 0=ekk, 1=d11, 2=d22, 3=d12, 4=d13, 5=d23
    auto ViewStrainOld() const { return Reshape(strain_old_packed.Read(), 6, ngllx_, nglly_, ngllz_, ne_); }
    auto ViewStrainOldWrite() { return Reshape(strain_old_packed.ReadWrite(), 6, ngllx_, nglly_, ngllz_, ne_); }

private:
    int ne_ = 0, ngllx_ = 0, nglly_ = 0, ngllz_ = 0, n_units_ = 0;
};


/**
 * @brief 2D Viscoacoustic memory state (Crank-Nicolson scheme)
 *
 * Stores memory variables and Crank-Nicolson coefficients for 2D viscoacoustic
 * computation. Only 1 memory variable per mechanism (bulk modulus relaxation).
 *
 * Memory update (Crank-Nicolson):
 *   e1_new = alpha × e1_old + strain_coeff × (strain_old + strain_new)
 *
 * Where strain = stiffness operator output = -∇·((1/ρ)∇φ)
 */
class ViscoacousticMemory2D {
public:
    ViscoacousticMemory2D() = default;
    ViscoacousticMemory2D(int ne, int ngllx, int nglly, int n_units);

    void Reset();
    bool IsInitialized() const { return n_units_ > 0; }
    int NumUnits() const { return n_units_; }
    int Ne() const { return ne_; }
    int Ngllx() const { return ngllx_; }
    int Nglly() const { return nglly_; }
    size_t MemoryUsage() const;

    /// Enable GPU memory for all vectors
    void EnableDevice() {
        e1.UseDevice(true);
        coeffs_packed.UseDevice(true);
        dot_e1_old.UseDevice(true);
    }

    /// Force host→device sync for all vectors
    void SyncToDevice() {
        e1.Read();
        coeffs_packed.Read();
        dot_e1_old.Read();
    }

    // =========================================================================
    // View Factory Methods
    // =========================================================================

    /// Memory variable view [ngllx, nglly, ne, n_units]
    auto ViewE1() { return Reshape(e1.ReadWrite(), ngllx_, nglly_, ne_, n_units_); }
    auto ViewE1Read() const { return Reshape(e1.Read(), ngllx_, nglly_, ne_, n_units_); }

    /// Old stiffness output view [ngllx, nglly, ne]
    auto ViewDotE1Old() { return Reshape(dot_e1_old.ReadWrite(), ngllx_, nglly_, ne_); }
    auto ViewDotE1OldRead() const { return Reshape(dot_e1_old.Read(), ngllx_, nglly_, ne_); }

    /// Crank-Nicolson coefficient view [2, ngllx, nglly, ne, n_units]
    /// Components: 0=alpha, 1=strain_coeff
    auto ViewCoeffs() const { return Reshape(coeffs_packed.Read(), 2, ngllx_, nglly_, ne_, n_units_); }

    // =========================================================================
    // Memory Arrays
    // =========================================================================

    /// Memory variable e1 [ngllx * nglly * ne * n_units]
    Vector e1;

    /// Crank-Nicolson coefficients [2, ngllx * nglly * ne * n_units]
    /// Components: 0=alpha, 1=strain_coeff
    Vector coeffs_packed;

    /// Old stiffness output dot_e1_old [ngllx * nglly * ne]
    Vector dot_e1_old;

private:
    int ne_ = 0, ngllx_ = 0, nglly_ = 0, n_units_ = 0;
};





/**
 * @brief 3D Viscoacoustic memory state (Crank-Nicolson scheme)
 *
 * Stores memory variables and Crank-Nicolson coefficients for 3D viscoacoustic
 * computation. Only 1 memory variable per mechanism (bulk modulus relaxation).
 *
 * Memory update (Crank-Nicolson):
 *   e1_new = alpha × e1_old + strain_coeff × (strain_old + strain_new)
 *
 * Where strain = stiffness operator output = -∇·((1/ρ)∇φ)
 */
class ViscoacousticMemory3D {
public:
    ViscoacousticMemory3D() = default;
    ViscoacousticMemory3D(int ne, int ngllx, int nglly, int ngllz, int n_units);

    void Reset();
    bool IsInitialized() const { return n_units_ > 0; }
    int NumUnits() const { return n_units_; }
    int Ne() const { return ne_; }
    int Ngllx() const { return ngllx_; }
    int Nglly() const { return nglly_; }
    int Ngllz() const { return ngllz_; }
    size_t MemoryUsage() const;

    /// Enable GPU memory for all vectors
    void EnableDevice() {
        e1.UseDevice(true);
        coeffs_packed.UseDevice(true);
        dot_e1_old.UseDevice(true);
    }

    /// Force host→device sync for all vectors
    void SyncToDevice() {
        e1.Read();
        coeffs_packed.Read();
        dot_e1_old.Read();
    }

    // =========================================================================
    // View Factory Methods
    // =========================================================================

    /// Memory variable view [ngllx, nglly, ngllz, ne, n_units]
    auto ViewE1() { return Reshape(e1.ReadWrite(), ngllx_, nglly_, ngllz_, ne_, n_units_); }
    auto ViewE1Read() const { return Reshape(e1.Read(), ngllx_, nglly_, ngllz_, ne_, n_units_); }

    /// Old stiffness output view [ngllx, nglly, ngllz, ne]
    auto ViewDotE1Old() { return Reshape(dot_e1_old.ReadWrite(), ngllx_, nglly_, ngllz_, ne_); }
    auto ViewDotE1OldRead() const { return Reshape(dot_e1_old.Read(), ngllx_, nglly_, ngllz_, ne_); }

    /// Crank-Nicolson coefficient view [2, ngllx, nglly, ngllz, ne, n_units]
    /// Components: 0=alpha, 1=strain_coeff
    auto ViewCoeffs() const { return Reshape(coeffs_packed.Read(), 2, ngllx_, nglly_, ngllz_, ne_, n_units_); }

    // =========================================================================
    // Memory Arrays
    // =========================================================================

    /// Memory variable e1 [ngllx * nglly * ngllz * ne * n_units]
    Vector e1;

    /// Crank-Nicolson coefficients [2, ngllx * nglly * ngllz * ne * n_units]
    /// Components: 0=alpha, 1=strain_coeff
    Vector coeffs_packed;

    /// Old stiffness output dot_e1_old [ngllx * nglly * ngllz * ne]
    Vector dot_e1_old;

private:
    int ne_ = 0, ngllx_ = 0, nglly_ = 0, ngllz_ = 0, n_units_ = 0;
};















}  // namespace SEM

#endif  // SEM_ATTENUATION_MODEL_HPP
