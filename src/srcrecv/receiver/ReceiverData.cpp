/**
 * @file ReceiverData.cpp
 * @brief ReceiverData class implementation (single receiver data management)
 */

#include "srcrecv/Receiver.hpp"

namespace SEM {

// =============================================================================
// ReceiverData
// =============================================================================

ReceiverData::ReceiverData(const std::string& name,
                           const Vector& position,
                           ReceiverType type,
                           int nt, real_t dt, int ncomp)
    : name_(name), position_(position), type_(type),
      nt_(nt), dt_(dt), ncomp_(ncomp)
{
    // data_ and weight_ are NOT allocated here.
    // They are allocated later only for local receivers via AllocateDataStorage().
    // This saves memory when there are many receivers across many MPI ranks.
}

void ReceiverData::SetSample(int step, const Vector& sample) {
    for (int c = 0; c < ncomp_; c++) {
        data_(step, c) = sample[c];
    }
}

void ReceiverData::AddSample(int step, const Vector& sample) {
    for (int c = 0; c < ncomp_; c++) {
        data_(step, c) += sample[c];
    }
}

void ReceiverData::ResetData() {
    data_ = 0.0;
}

void ReceiverData::Resize(int nt, real_t dt) {
    nt_ = nt;
    dt_ = dt;
    // Only resize if data storage is already allocated (local receivers)
    if (data_.NumRows() > 0) {
        data_.SetSize(nt, ncomp_);
        data_ = 0.0;
        weight_.SetSize(nt, ncomp_);
        weight_ = 1.0;
    }
}

void ReceiverData::AllocateDataStorage() {
    if (data_.NumRows() == 0 && nt_ > 0 && ncomp_ > 0) {
        data_.SetSize(nt_, ncomp_);
        data_ = 0.0;
        weight_.SetSize(nt_, ncomp_);
        weight_ = initial_weight_;  // Use stored weight value
    }
}

bool ReceiverData::HasDataStorage() const {
    return data_.NumRows() > 0;
}

void ReceiverData::SetLocation(int elem, int owner_rank, const Vector& ref_pos, bool is_local) {
    elem_ = elem;
    owner_rank_ = owner_rank;
    ref_position_ = ref_pos;
    is_local_ = is_local;
}

void ReceiverData::SetWeight(real_t w) {
    initial_weight_ = w;
    if (weight_.NumRows() > 0) {
        weight_ = w;  // Apply immediately if already allocated
    }
}

void ReceiverData::SetWeight(const DenseMatrix& w) {
    weight_ = w;
}

void ReceiverData::CacheInterpolation(const Vector& shape, const Array<int>& vdofs) {
    cached_shape_ = shape;
    cached_vdofs_ = vdofs;

    // Enable GPU memory and force host→device sync
    // This allows GetSubVector to use GPU path instead of DtoH transfer
    cached_shape_.UseDevice(true);
    cached_shape_.Read();  // Force sync to GPU

    cached_vdofs_.GetMemory().UseDevice(true);
    cached_vdofs_.Read();  // Force sync to GPU
}

}  // namespace SEM
