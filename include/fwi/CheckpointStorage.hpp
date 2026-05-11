/**
 * @file CheckpointStorage.hpp
 * @brief Checkpoint storage for Revolve-based adjoint computation
 *
 * Stores simulation state (u, v, a + optional attenuation memory variables)
 * for forward replay during adjoint time stepping.
 *
 * Two storage backends:
 *   - InMemoryStorage: In-memory (host RAM or device VRAM) — fast, limited by RAM
 *   - DiskStorage: On-disk — slow, unlimited capacity (future)
 */

#ifndef SEM_CHECKPOINT_STORAGE_HPP
#define SEM_CHECKPOINT_STORAGE_HPP

#include <mfem.hpp>
#include <vector>
#include <string>

namespace SEM {

using mfem::real_t;
using mfem::Vector;

/**
 * @brief Single checkpoint: u, v, a + optional attenuation state
 */
struct Checkpoint {
    Vector u;                   // Displacement
    Vector v;                   // Velocity
    Vector a;                   // Acceleration
    Vector attenuation_state;   // Packed attenuation memory (empty if non-visco)
    int step = -1;              // Time step number
    bool valid = false;         // Whether this slot contains data
};

/**
 * @brief Abstract base class for checkpoint storage
 */
class CheckpointStorage {
public:
    virtual ~CheckpointStorage() = default;

    /**
     * @brief Save state to a checkpoint slot
     * @param slot Slot index (0 to num_slots-1)
     * @param u Displacement vector
     * @param v Velocity vector
     * @param a Acceleration vector
     * @param attenuation_state Packed attenuation state (empty if non-visco)
     * @param step Time step number
     */
    virtual void Save(int slot, const Vector& u, const Vector& v,
                      const Vector& a, const Vector& attenuation_state,
                      int step) = 0;

    /**
     * @brief Load state from a checkpoint slot
     * @param slot Slot index
     * @param u Output displacement
     * @param v Output velocity
     * @param a Output acceleration
     * @param attenuation_state Output attenuation state
     * @param step Output time step number
     */
    virtual void Load(int slot, Vector& u, Vector& v, Vector& a,
                      Vector& attenuation_state, int& step) = 0;

    /// Number of available slots
    virtual int NumSlots() const = 0;

    /// Check if a slot contains valid data
    virtual bool IsValid(int slot) const = 0;

    /// Total memory usage in bytes
    virtual size_t MemoryUsage() const = 0;
};

/**
 * @brief In-memory checkpoint storage
 *
 * Pre-allocates all slots at construction time.
 * Data placement (host RAM vs device VRAM) follows MFEM Vector semantics.
 */
class InMemoryStorage : public CheckpointStorage {
public:
    /**
     * @brief Construct with given number of slots
     * @param num_slots Number of checkpoint slots to allocate
     */
    explicit InMemoryStorage(int num_slots);

    void Save(int slot, const Vector& u, const Vector& v,
              const Vector& a, const Vector& attenuation_state,
              int step) override;

    void Load(int slot, Vector& u, Vector& v, Vector& a,
              Vector& attenuation_state, int& step) override;

    int NumSlots() const override { return static_cast<int>(slots_.size()); }
    bool IsValid(int slot) const override;
    size_t MemoryUsage() const override;

private:
    std::vector<Checkpoint> slots_;
};

}  // namespace SEM

#endif  // SEM_CHECKPOINT_STORAGE_HPP
