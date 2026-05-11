/**
 * @file SimulationComponents.cpp
 * @brief Implementation of SimulationComponents template class
 */

#include "simulation/SimulationComponents.hpp"

namespace SEM {

// =============================================================================
// Constructor / Destructor
// =============================================================================

template<int Dim>
SimulationComponents<Dim>::SimulationComponents(MPI_Comm comm)
    : comm_(comm)
{
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &num_procs_);
}

template<int Dim>
SimulationComponents<Dim>::~SimulationComponents() = default;

// =============================================================================
// Mesh Management
// =============================================================================

template<int Dim>
void SimulationComponents<Dim>::SetMesh(std::unique_ptr<ParMesh> mesh) {
    MFEM_VERIFY(mesh != nullptr, "Cannot set null mesh");
    MFEM_VERIFY(mesh->Dimension() == Dim,
                "Mesh dimension " << mesh->Dimension() << " != template Dim " << Dim);
    mesh_ = std::move(mesh);
}

// =============================================================================
// FE Space Management
// =============================================================================

template<int Dim>
void SimulationComponents<Dim>::CreateFESpaces(int order, int vdim) {
    MFEM_VERIFY(mesh_ != nullptr, "Mesh must be set before creating FE spaces");
    MFEM_VERIFY(order > 0, "Order must be positive");
    MFEM_VERIFY(vdim > 0, "Vector dimension must be positive");

    order_ = order;
    vdim_ = vdim;

    // Create FE collection (H1 continuous elements)
    fec_ = std::make_unique<H1_FECollection>(order, Dim);

    // Create vector FE space (vdim components)
    fes_ = std::make_unique<ParFiniteElementSpace>(mesh_.get(), fec_.get(), vdim);

    // Create scalar FE space (for materials, etc.)
    fes_scalar_ = std::make_unique<ParFiniteElementSpace>(mesh_.get(), fec_.get(), 1);

    // Create state vectors
    u_ = std::make_unique<ParGridFunction>(fes_.get());
    v_ = std::make_unique<ParGridFunction>(fes_.get());
    a_ = std::make_unique<ParGridFunction>(fes_.get());

    // Enable GPU device memory for state vectors
    // This allows mfem::forall kernels to access data on GPU
    u_->UseDevice(true);
    v_->UseDevice(true);
    a_->UseDevice(true);

    // Initialize to zero
    *u_ = 0.0;
    *v_ = 0.0;
    *a_ = 0.0;
}

// =============================================================================
// State Vector Operations
// =============================================================================

template<int Dim>
void SimulationComponents<Dim>::ResetState() {
    if (u_) *u_ = 0.0;
    if (v_) *v_ = 0.0;
    if (a_) *a_ = 0.0;
}

// =============================================================================
// Memory Usage
// =============================================================================

template<int Dim>
size_t SimulationComponents<Dim>::MemoryUsage() const {
    size_t total = 0;

    if (u_) total += u_->Size() * sizeof(real_t);
    if (v_) total += v_->Size() * sizeof(real_t);
    if (a_) total += a_->Size() * sizeof(real_t);

    // Mesh memory is harder to estimate accurately
    // This is a rough estimate
    if (mesh_) {
        total += mesh_->GetNE() * sizeof(int) * 10;  // Elements
        total += mesh_->GetNV() * sizeof(real_t) * Dim;  // Vertices
    }

    return total;
}

// =============================================================================
// Explicit Template Instantiations
// =============================================================================

template class SimulationComponents<2>;
template class SimulationComponents<3>;

}  // namespace SEM
