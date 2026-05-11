/**
 * @file SimulationComponents.hpp
 * @brief Core simulation components: mesh, FE spaces, and state vectors
 *
 * SimulationComponents owns and manages:
 * - ParMesh (parallel mesh)
 * - FiniteElementCollection and ParFiniteElementSpace
 * - State vectors (displacement u, velocity v, acceleration a)
 *
 * This class is designed to be composed into SimulationFacade.
 */

#ifndef SEM_SIMULATION_COMPONENTS_HPP
#define SEM_SIMULATION_COMPONENTS_HPP

#include <mfem.hpp>
#include <memory>

namespace SEM {


using namespace mfem;

/**
 * @class SimulationComponents
 * @brief Manages mesh, FE spaces, and state vectors for simulation
 * @tparam Dim Spatial dimension (2 or 3)
 *
 * Responsibilities:
 * - Mesh ownership and access
 * - FE collection and space creation
 * - State vector (u, v, a) ownership and initialization
 *
 * This class does NOT handle:
 * - Time integration (see SimulationRunner)
 * - Output (see SimulationIO)
 * - Physics/operators (owned by SimulationFacade)
 */
template<int Dim>
class SimulationComponents {
public:
    /**
     * @brief Construct components with MPI communicator
     * @param comm MPI communicator
     */
    explicit SimulationComponents(MPI_Comm comm);

    /// Destructor
    ~SimulationComponents();

    // Prevent copying (owns MFEM objects)
    SimulationComponents(const SimulationComponents&) = delete;
    SimulationComponents& operator=(const SimulationComponents&) = delete;

    // Allow moving
    SimulationComponents(SimulationComponents&&) = default;
    SimulationComponents& operator=(SimulationComponents&&) = default;

    // -------------------------------------------------------------------------
    // Mesh Management
    // -------------------------------------------------------------------------

    /**
     * @brief Set the parallel mesh
     * @param mesh ParMesh (takes ownership)
     */
    void SetMesh(std::unique_ptr<ParMesh> mesh);

    /// Get mesh reference
    ParMesh& Mesh() { return *mesh_; }
    const ParMesh& Mesh() const { return *mesh_; }

    /// Check if mesh is set
    bool HasMesh() const { return mesh_ != nullptr; }

    // -------------------------------------------------------------------------
    // FE Space Management
    // -------------------------------------------------------------------------

    /**
     * @brief Create FE spaces after mesh is set
     * @param order Polynomial order
     * @param vdim Vector dimension (Dim for elastic, 1 for acoustic)
     *
     * Creates:
     * - fec_: H1_FECollection of given order
     * - fes_: Vector FE space with vdim components
     * - fes_scalar_: Scalar FE space (vdim=1)
     * - u_, v_, a_: State vectors on fes_
     */
    void CreateFESpaces(int order, int vdim);

    /// Get vector FE space
    ParFiniteElementSpace& FES() { return *fes_; }
    const ParFiniteElementSpace& FES() const { return *fes_; }

    /// Get scalar FE space (for materials, etc.)
    ParFiniteElementSpace& FESScalar() { return *fes_scalar_; }
    const ParFiniteElementSpace& FESScalar() const { return *fes_scalar_; }

    /// Get FE collection
    const FiniteElementCollection& FECollection() const { return *fec_; }

    /// Check if FE spaces are created
    bool HasFESpaces() const { return fes_ != nullptr; }

    // -------------------------------------------------------------------------
    // State Vectors
    // -------------------------------------------------------------------------

    /// Displacement field
    ParGridFunction& U() { return *u_; }
    const ParGridFunction& U() const { return *u_; }

    /// Velocity field
    ParGridFunction& V() { return *v_; }
    const ParGridFunction& V() const { return *v_; }

    /// Acceleration field
    ParGridFunction& A() { return *a_; }
    const ParGridFunction& A() const { return *a_; }

    /// Reset state vectors to zero
    void ResetState();

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /// Set polynomial order (before CreateFESpaces)
    void SetOrder(int order) { order_ = order; }

    // -------------------------------------------------------------------------
    // Query
    // -------------------------------------------------------------------------

    /// Get polynomial order
    int Order() const { return order_; }

    /// Get vector dimension
    int VDim() const { return vdim_; }

    /// Get MPI communicator
    MPI_Comm Comm() const { return comm_; }

    /// Get MPI rank
    int Rank() const { return rank_; }

    /// Check if root process
    bool IsRoot() const { return rank_ == 0; }

    /// Get number of local elements
    int NumLocalElements() const { return mesh_ ? mesh_->GetNE() : 0; }

    /// Get total DOFs (local)
    int NumDOFs() const { return fes_ ? fes_->GetTrueVSize() : 0; }

    /// Get global DOFs (MPI collective)
    HYPRE_BigInt GlobalNumDOFs() const { return fes_ ? fes_->GlobalTrueVSize() : 0; }

    /// Estimate memory usage (bytes)
    size_t MemoryUsage() const;

private:
    MPI_Comm comm_;
    int rank_ = 0;
    int num_procs_ = 1;

    // Order and dimension
    int order_ = 0;
    int vdim_ = 0;

    // Mesh (owned)
    std::unique_ptr<ParMesh> mesh_;

    // FE collection and spaces (owned)
    std::unique_ptr<FiniteElementCollection> fec_;
    std::unique_ptr<ParFiniteElementSpace> fes_;
    std::unique_ptr<ParFiniteElementSpace> fes_scalar_;

    // State vectors (owned)
    std::unique_ptr<ParGridFunction> u_;
    std::unique_ptr<ParGridFunction> v_;
    std::unique_ptr<ParGridFunction> a_;
};

// Type aliases
using SimulationComponents2D = SimulationComponents<2>;
using SimulationComponents3D = SimulationComponents<3>;

}  // namespace SEM

#endif  // SEM_SIMULATION_COMPONENTS_HPP
