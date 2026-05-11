/**
 * @file GeneralizedZener.hpp
 * @brief Generalized Zener (Standard Linear Solid) attenuation model
 *
 * Implements viscoelastic attenuation using the Generalized Zener model
 * with multiple relaxation mechanisms. Each mechanism has its own
 * relaxation time and modulus defect.
 *
 * Memory variable update (Crank-Nicolson):
 *   M_new = A * M_old + B * strain_old + gamma * strain_new
 * where:
 *   A = exp(-dt/tau) (decay factor)
 *   B, gamma = coefficients from time integration scheme
 *
 * 2D: 3 memory variables per mechanism
 *   - M1: bulk (volumetric) for tr(ε)
 *   - M2: deviatoric d11 = εxx - εkk/2
 *   - M3: shear d12 = εxy
 *
 * 3D: 6 memory variables per mechanism
 *   - M1: bulk (volumetric) for tr(ε)
 *   - M2: deviatoric d11 = εxx - εkk/3
 *   - M3: deviatoric d22 = εyy - εkk/3
 *   - M4: shear εxy
 *   - M5: shear εxz
 *   - M6: shear εyz
 */

#ifndef SEM_GENERALIZED_ZENER_HPP
#define SEM_GENERALIZED_ZENER_HPP

#include "integ/attenuation/AttenuationModel.hpp"
#include "AttenuationCoeffs.hpp"
#include <memory>

namespace SEM {

/**
 * @class GeneralizedZener2D
 * @brief 2D Generalized Zener viscoelastic attenuation
 */
class GeneralizedZener2D {
public:
    static constexpr int NUM_MEM_VARS = 3;  // bulk + 2 deviatoric
    static constexpr bool HAS_MEMORY = true;
    static constexpr int DIM = 2;

    GeneralizedZener2D() = default;

    /// Number of memory variables per mechanism
    int NumMemVarsPerMechanism() const { return NUM_MEM_VARS; }

    /// Check if model has memory variables
    bool HasMemory() const { return true; }

    /// Get memory state
    ViscoelasticMemory2D* GetMemory() { return memory_.get(); }
    const ViscoelasticMemory2D* GetMemory() const { return memory_.get(); }

    /// Initialize memory state
    void InitState(int ne, int ngllx, int nglly, int n_mechanisms) {
        memory_ = std::make_unique<ViscoelasticMemory2D>(ne, ngllx, nglly, n_mechanisms);
    }

    /// Reset memory variables to zero
    void Reset() {
        if (memory_) {
            memory_->Reset();
        }
    }

    /// Check if initialized
    bool IsInitialized() const { return memory_ != nullptr && memory_->IsInitialized(); }

    /// Number of SLS mechanisms
    int NumMechanisms() const { return memory_ ? memory_->NumUnits() : 0; }

    /// Memory usage in bytes
    size_t MemoryUsage() const { return memory_ ? memory_->MemoryUsage() : 0; }

    /**
     * @brief Enable attenuation with Q factors
     *
     * Computes relaxation parameters from target Q values and initializes
     * memory state. This is the main entry point for enabling viscoelasticity.
     *
     * Also applies the unrelaxed modulus correction (physical dispersion)
     * to kappa and mu vectors in-place.
     *
     * @param ne Number of elements
     * @param ngllx, nglly Number of GLL points per direction
     * @param Qkappa Q factor for bulk modulus at each GLL point
     * @param Qmu Q factor for shear modulus at each GLL point
     * @param kappa Bulk modulus at each GLL point (will be corrected in-place)
     * @param mu Shear modulus at each GLL point (will be corrected in-place)
     * @param f0 Reference frequency (Hz)
     * @param n_sls Number of SLS relaxation mechanisms
     * @param dt Time step for memory variable update
     */
    void EnableAttenuation(int ne, int ngllx, int nglly,
                           const Vector& Qkappa, const Vector& Qmu,
                           Vector& kappa, Vector& mu,
                           real_t f0, int n_sls, real_t dt);

    /**
     * @brief Update memory variables (Newmark scheme)
     *
     * Called during stiffness kernel to update memory state.
     * M_new = A * M_old + B * strain_new
     *
     * @param q GLL point index
     * @param eps_xx, eps_yy, eps_xy Current strain components
     */
    void UpdateMemoryAtPoint(int q, real_t eps_xx, real_t eps_yy, real_t eps_xy);

    /**
     * @brief Get attenuation stress correction at a point
     *
     * Returns the stress correction from memory variables:
     *   sigma_att = sum_l(Phi_l * M_l)
     *
     * @param q GLL point index
     * @param sig_att_xx, sig_att_yy, sig_att_xy Output stress corrections
     */
    void GetAttenuationStressAtPoint(int q,
        real_t& sig_att_xx, real_t& sig_att_yy, real_t& sig_att_xy) const;

private:
    std::unique_ptr<ViscoelasticMemory2D> memory_;
    real_t dt_ = 0.0;
};


/**
 * @class GeneralizedZener3D
 * @brief 3D Generalized Zener viscoelastic attenuation
 */
class GeneralizedZener3D {
public:
    static constexpr int NUM_MEM_VARS = 6;  // bulk + 5 deviatoric/shear
    static constexpr bool HAS_MEMORY = true;
    static constexpr int DIM = 3;

    GeneralizedZener3D() = default;

    /// Number of memory variables per mechanism
    int NumMemVarsPerMechanism() const { return NUM_MEM_VARS; }

    /// Check if model has memory variables
    bool HasMemory() const { return true; }

    /// Get memory state
    ViscoelasticMemory3D* GetMemory() { return memory_.get(); }
    const ViscoelasticMemory3D* GetMemory() const { return memory_.get(); }

    /// Initialize memory state
    void InitState(int ne, int ngllx, int nglly, int ngllz, int n_mechanisms) {
        memory_ = std::make_unique<ViscoelasticMemory3D>(ne, ngllx, nglly, ngllz, n_mechanisms);
    }

    /// Reset memory variables to zero
    void Reset() {
        if (memory_) {
            memory_->Reset();
        }
    }

    /// Check if initialized
    bool IsInitialized() const { return memory_ != nullptr && memory_->IsInitialized(); }

    /// Number of SLS mechanisms
    int NumMechanisms() const { return memory_ ? memory_->NumUnits() : 0; }

    /// Memory usage in bytes
    size_t MemoryUsage() const { return memory_ ? memory_->MemoryUsage() : 0; }

    /**
     * @brief Enable attenuation with Q factors
     *
     * Computes relaxation parameters from target Q values and initializes
     * memory state. Uses Crank-Nicolson scheme for memory variable update.
     *
     * Also applies the unrelaxed modulus correction (physical dispersion)
     * to kappa and mu vectors in-place.
     *
     * @param ne Number of elements
     * @param ngllx, nglly, ngllz Number of GLL points per direction
     * @param Qkappa Q factor for bulk modulus at each GLL point
     * @param Qmu Q factor for shear modulus at each GLL point
     * @param kappa Bulk modulus at each GLL point (will be corrected in-place)
     * @param mu Shear modulus at each GLL point (will be corrected in-place)
     * @param f0 Reference frequency (Hz)
     * @param n_sls Number of SLS relaxation mechanisms
     * @param dt Time step for memory variable update
     */
    void EnableAttenuation(int ne, int ngllx, int nglly, int ngllz,
                           const Vector& Qkappa, const Vector& Qmu,
                           Vector& kappa, Vector& mu,
                           real_t f0, int n_sls, real_t dt);

    /**
     * @brief Update memory variables (Crank-Nicolson scheme)
     *
     * @param q GLL point index
     * @param eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz Current strain components
     */
    void UpdateMemoryAtPoint(int q,
        real_t eps_xx, real_t eps_yy, real_t eps_zz,
        real_t eps_xy, real_t eps_xz, real_t eps_yz);

    /**
     * @brief Get attenuation stress correction at a point
     *
     * @param q GLL point index
     * @param sig_att_xx, sig_att_yy, sig_att_zz, sig_att_xy, sig_att_xz, sig_att_yz Output
     */
    void GetAttenuationStressAtPoint(int q,
        real_t& sig_att_xx, real_t& sig_att_yy, real_t& sig_att_zz,
        real_t& sig_att_xy, real_t& sig_att_xz, real_t& sig_att_yz) const;

private:
    std::unique_ptr<ViscoelasticMemory3D> memory_;
    real_t dt_ = 0.0;
};


/**
 * @class GeneralizedZenerAcoustic2D
 * @brief 2D Generalized Zener viscoacoustic attenuation (Crank-Nicolson scheme)
 *
 * Implements viscoacoustic attenuation using the Generalized Zener model.
 * Uses Crank-Nicolson scheme matching 2D/3D viscoelastic implementation.
 *
 * Memory variable update (Crank-Nicolson):
 *   e1_new = alpha * e1_old + strain_coeff * (strain_old + strain_new)
 *
 * Where strain = stiffness operator output = -∇·((1/ρ)∇φ)
 *
 * Only 1 memory variable per mechanism (bulk modulus relaxation only, no shear).
 */
class GeneralizedZenerAcoustic2D {
public:
    static constexpr int NUM_MEM_VARS = 1;  // bulk only (no shear in acoustic)
    static constexpr bool HAS_MEMORY = true;
    static constexpr int DIM = 2;

    GeneralizedZenerAcoustic2D() = default;

    /// Number of memory variables per mechanism
    int NumMemVarsPerMechanism() const { return NUM_MEM_VARS; }

    /// Check if model has memory variables
    bool HasMemory() const { return true; }

    /// Get memory state
    ViscoacousticMemory2D* GetMemory() { return memory_.get(); }
    const ViscoacousticMemory2D* GetMemory() const { return memory_.get(); }

    /// Check if attenuation is enabled
    bool HasAttenuation() const { return memory_ && memory_->IsInitialized(); }

    /// Reset memory variables to zero
    void Reset() {
        if (memory_) {
            memory_->Reset();
        }
    }

    /// Check if initialized
    bool IsInitialized() const { return memory_ != nullptr && memory_->IsInitialized(); }

    /// Number of SLS mechanisms
    int NumMechanisms() const { return memory_ ? memory_->NumUnits() : 0; }

    /// Memory usage in bytes
    size_t MemoryUsage() const { return memory_ ? memory_->MemoryUsage() : 0; }

    /**
     * @brief Enable viscoacoustic attenuation with Qkappa
     *
     * Computes Crank-Nicolson coefficients from target Q values and initializes
     * memory state. Uses same scheme as 2D/3D viscoelastic implementation.
     *
     * Also applies the unrelaxed modulus correction (physical dispersion)
     * to kappa vector in-place.
     *
     * @param ne Number of elements
     * @param ngllx, nglly Number of GLL points per direction
     * @param Qkappa Q factor for bulk modulus at each GLL point
     * @param kappa Bulk modulus at each GLL point (will be corrected in-place)
     * @param f0 Reference frequency (Hz)
     * @param n_sls Number of SLS relaxation mechanisms
     * @param dt Time step for memory variable update
     */
    void EnableAttenuation(int ne, int ngllx, int nglly,
                           const Vector& Qkappa,
                           Vector& kappa,
                           real_t f0, int n_sls, real_t dt);

private:
    std::unique_ptr<ViscoacousticMemory2D> memory_;
    real_t dt_ = 0.0;
};




/**
 * @class GeneralizedZenerAcoustic3D
 * @brief 3D Generalized Zener viscoacoustic attenuation (Crank-Nicolson scheme)
 *
 * Implements viscoacoustic attenuation using the Generalized Zener model.
 * Uses Crank-Nicolson scheme matching 2D/3D viscoelastic implementation.
 *
 * Memory variable update (Crank-Nicolson):
 *   e1_new = alpha * e1_old + strain_coeff * (strain_old + strain_new)
 *
 * Where strain = stiffness operator output = -∇·((1/ρ)∇φ)
 *
 * Only 1 memory variable per mechanism (bulk modulus relaxation only, no shear).
 */
class GeneralizedZenerAcoustic3D {
public:
    static constexpr int NUM_MEM_VARS = 1;  // bulk only (no shear in acoustic)
    static constexpr bool HAS_MEMORY = true;
    static constexpr int DIM = 3;

    GeneralizedZenerAcoustic3D() = default;

    /// Number of memory variables per mechanism
    int NumMemVarsPerMechanism() const { return NUM_MEM_VARS; }

    /// Check if model has memory variables
    bool HasMemory() const { return true; }

    /// Get memory state
    ViscoacousticMemory3D* GetMemory() { return memory_.get(); }
    const ViscoacousticMemory3D* GetMemory() const { return memory_.get(); }

    /// Check if attenuation is enabled
    bool HasAttenuation() const { return memory_ && memory_->IsInitialized(); }

    /// Reset memory variables to zero
    void Reset() {
        if (memory_) {
            memory_->Reset();
        }
    }

    /// Check if initialized
    bool IsInitialized() const { return memory_ != nullptr && memory_->IsInitialized(); }

    /// Number of SLS mechanisms
    int NumMechanisms() const { return memory_ ? memory_->NumUnits() : 0; }

    /// Memory usage in bytes
    size_t MemoryUsage() const { return memory_ ? memory_->MemoryUsage() : 0; }

    /**
     * @brief Enable viscoacoustic attenuation with Qkappa
     *
     * Computes Crank-Nicolson coefficients from target Q values and initializes
     * memory state. Uses same scheme as 2D/3D viscoelastic implementation.
     *
     * Also applies the unrelaxed modulus correction (physical dispersion)
     * to kappa vector in-place.
     *
     * @param ne Number of elements
     * @param ngllx, nglly, ngllz Number of GLL points per direction
     * @param Qkappa Q factor for bulk modulus at each GLL point
     * @param kappa Bulk modulus at each GLL point (will be corrected in-place)
     * @param f0 Reference frequency (Hz)
     * @param n_sls Number of SLS relaxation mechanisms
     * @param dt Time step for memory variable update
     */
    void EnableAttenuation(int ne, int ngllx, int nglly, int ngllz,
                           const Vector& Qkappa,
                           Vector& kappa,
                           real_t f0, int n_sls, real_t dt);

private:
    std::unique_ptr<ViscoacousticMemory3D> memory_;
    real_t dt_ = 0.0;
};
















}  // namespace SEM

#endif  // SEM_GENERALIZED_ZENER_HPP
