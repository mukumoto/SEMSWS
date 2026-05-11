/**
 * @file CheckpointStorage.cpp
 * @brief Checkpoint storage implementations
 */

#include "fwi/CheckpointStorage.hpp"

namespace SEM {

// =============================================================================
// InMemoryStorage
// =============================================================================

InMemoryStorage::InMemoryStorage(int num_slots)
    : slots_(num_slots)
{
}

void InMemoryStorage::Save(int slot, const Vector& u, const Vector& v,
                              const Vector& a, const Vector& attenuation_state,
                              int step)
{
    MFEM_VERIFY(slot >= 0 && slot < static_cast<int>(slots_.size()),
        "Checkpoint slot " << slot << " out of range [0, " << slots_.size() << ")");

    Checkpoint& cp = slots_[slot];

    cp.u = u;
    cp.v = v;
    cp.a = a;

    if (attenuation_state.Size() > 0) {
        cp.attenuation_state = attenuation_state;
    } else {
        cp.attenuation_state.SetSize(0);
    }

    cp.step = step;
    cp.valid = true;
}

void InMemoryStorage::Load(int slot, Vector& u, Vector& v, Vector& a,
                              Vector& attenuation_state, int& step)
{
    MFEM_VERIFY(slot >= 0 && slot < static_cast<int>(slots_.size()),
        "Checkpoint slot " << slot << " out of range [0, " << slots_.size() << ")");
    MFEM_VERIFY(slots_[slot].valid,
        "Checkpoint slot " << slot << " is not valid (not saved yet)");

    const Checkpoint& cp = slots_[slot];

    u = cp.u;
    v = cp.v;
    a = cp.a;

    if (cp.attenuation_state.Size() > 0) {
        attenuation_state = cp.attenuation_state;
    } else {
        attenuation_state.SetSize(0);
    }

    step = cp.step;
}

bool InMemoryStorage::IsValid(int slot) const {
    if (slot < 0 || slot >= static_cast<int>(slots_.size())) return false;
    return slots_[slot].valid;
}

size_t InMemoryStorage::MemoryUsage() const {
    size_t total = 0;
    for (const auto& cp : slots_) {
        total += cp.u.Size() * sizeof(real_t);
        total += cp.v.Size() * sizeof(real_t);
        total += cp.a.Size() * sizeof(real_t);
        total += cp.attenuation_state.Size() * sizeof(real_t);
    }
    return total;
}

}  // namespace SEM
