/**
 * @file ReceiverFactory.cpp
 * @brief ReceiverArray constructor, AddReceiver, Setup, FromConfig
 */

#include "srcrecv/Receiver.hpp"
#include "config/ConfigTypes.hpp"
#include "util/Profiler.hpp"

namespace SEM {

// StringToReceiverType is now in common/Types.hpp (included via Receiver.hpp)

// =============================================================================
// ReceiverArray - Construction and Setup
// =============================================================================

ReceiverArray::ReceiverArray(const ParFiniteElementSpace& fes,
                             MPI_Comm* comm,
                             DomainType domain)
    : fes_(fes), comm_(comm), domain_(domain)
{
    MPI_Comm_rank(*comm_, &local_rank_);
    MPI_Comm_size(*comm_, &num_procs_);
    space_dim_ = fes_.GetParMesh()->SpaceDimension();
}

void ReceiverArray::LoadFromTextFile(const std::string& filepath, int nt, real_t dt) {
    Init();
    nt_ = nt;
    dt_ = dt;

    ParseTextFile(filepath);

    // Assign global IDs to all receivers
    int id = 0;
    for (auto& entry : receivers_) {
        for (auto& rec : entry.second) {
            rec.SetId(id++);
        }
    }
    num_total_ = id;

    LocateReceivers();
    PrecomputeInterpolation();  // Cache shape functions and DOFs for efficiency

    // Allocate data storage only for local receivers
    for (auto& entry : receivers_) {
        for (auto& rec : entry.second) {
            if (rec.IsLocal()) {
                rec.AllocateDataStorage();
            }
        }
    }
}

void ReceiverArray::AddReceiver(const std::string& name,
                                 const Vector& position,
                                 ReceiverType type,
                                 real_t weight) {
    // Check domain type compatibility - filter incompatible types
    bool is_solid_receiver = (static_cast<int>(type) < 10);

    if (domain_ == DomainType::Solid && !is_solid_receiver) {
        // Log filtered receiver (acoustic type in elastic simulation)
        filtered_log_.push_back({name, ReceiverTypeToString(type),
            "acoustic receiver type in elastic domain"});
        return;  // Skip this receiver
    }
    if (domain_ == DomainType::Fluid && is_solid_receiver) {
        // Log filtered receiver (elastic type in acoustic simulation)
        filtered_log_.push_back({name, ReceiverTypeToString(type),
            "elastic receiver type in acoustic domain"});
        return;  // Skip this receiver
    }

    // Determine number of components
    int ncomp = 1;
    if (type == ReceiverType::Displacement ||
        type == ReceiverType::Velocity ||
        type == ReceiverType::Acceleration) {
        ncomp = space_dim_;
    } else if (type == ReceiverType::Gradient) {
        ncomp = space_dim_ * space_dim_;
    }

    // Create receiver (nt_/dt_ will be set in Setup())
    ReceiverData rec(name, position, type, 1, 0.0, ncomp);
    rec.SetWeight(weight);

    receivers_[type].push_back(std::move(rec));
    num_total_++;
}

void ReceiverArray::Setup(int nt, real_t dt) {
    PROFILE_REGION("Setup:ReceiverSetup");

    nt_ = nt;
    dt_ = dt;

    // Assign global IDs to all receivers (before LocateReceivers)
    int id = 0;
    for (auto& entry : receivers_) {
        for (auto& rec : entry.second) {
            rec.SetId(id++);
        }
    }

    {
        PROFILE_REGION("Setup:LocateReceivers");
        LocateReceivers();
    }
    {
        PROFILE_REGION("Setup:PrecomputeInterpolation");
        PrecomputeInterpolation();
    }

    // Update nt/dt and allocate data storage only for local receivers (after LocateReceivers)
    for (auto& entry : receivers_) {
        for (auto& rec : entry.second) {
            rec.Resize(nt, dt);  // Update nt_ and dt_ in each receiver
            if (rec.IsLocal()) {
                rec.AllocateDataStorage();
            }
        }
    }
}

std::map<ReceiverType, std::vector<ReceiverData>>& ReceiverArray::GetAllReceivers() {
    return receivers_;
}

const std::map<ReceiverType, std::vector<ReceiverData>>& ReceiverArray::GetAllReceivers() const {
    return receivers_;
}

void ReceiverArray::ResetData() {
    for (auto& entry : receivers_) {
        for (auto& rec : entry.second) {
            if (rec.HasDataStorage()) {
                rec.ResetData();
            }
        }
    }

    // Reset GPU recording buffers
    // After FlushDeviceBuffer, both host and device are marked valid.
    // RecordDevice uses ReadWrite() which doesn't invalidate host.
    // So the next FlushDeviceBuffer's HostRead() returns stale host data
    // instead of copying fresh device data. Fix: zero the buffer on device
    // (which invalidates host via Write()), and reset position counters.
    for (auto& [type, dd] : device_data_) {
        dd.d_buffer = 0.0;  // Zeros on device (UseDevice is true), invalidates host
        dd.current_pos = 0;
        dd.start_step = 0;
    }
}

void ReceiverArray::Init() {
    for (auto& entry : receivers_) {
        entry.second.clear();
    }
    receivers_.clear();
    num_total_ = 0;
    output_sources_.clear();
}

// =============================================================================
// ReceiverArray - FromConfig Factory
// =============================================================================

std::unique_ptr<ReceiverArray> ReceiverArray::FromConfig(
    const ReceiverConfig::Config& config,
    const ParFiniteElementSpace& fes,
    int nt, real_t dt,
    MPI_Comm* comm,
    DomainType domain) {

    auto array = std::make_unique<ReceiverArray>(fes, comm, domain);

    int dim = fes.GetParMesh()->SpaceDimension();

    for (size_t i = 0; i < config.receivers.size(); i++) {
        const auto& rec = config.receivers[i];
        Vector pos(dim);
        for (int d = 0; d < dim; d++) {
            pos[d] = rec.location[d];
        }

        // AddReceiver for each type
        for (const auto& type_str : rec.types) {
            ReceiverType rtype = StringToReceiverType(type_str);
            array->AddReceiver(rec.name, pos, rtype, rec.weight);
        }
    }

    array->Setup(nt, dt);
    return array;
}

}  // namespace SEM
