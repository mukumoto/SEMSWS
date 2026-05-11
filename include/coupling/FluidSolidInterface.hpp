// FluidSolidInterface — per-GLL bookkeeping for Γ_fs plus gather/scatter
// helpers that feed the φ − u_s coupling term in CoupledSimulationFacade.
// Owns the auto-detected boundary attribute, per-GLL normal and J·w, and
// DOF indices into the fluid scalar / solid vector FES. Local and
// remote-MPI paths run on host or device.

#ifndef SEM_COUPLING_FLUID_SOLID_INTERFACE_HPP
#define SEM_COUPLING_FLUID_SOLID_INTERFACE_HPP

#include <mfem.hpp>
#include <memory>
#include <vector>

namespace SEM {

using mfem::Array;
using mfem::ParFiniteElementSpace;
using mfem::ParGridFunction;
using mfem::ParMesh;
using mfem::ParSubMesh;
using mfem::Vector;
using mfem::real_t;

/**
 * @class FluidSolidInterface
 * @tparam Dim spatial dimension (2 or 3)
 */
template<int Dim>
class FluidSolidInterface {
public:
    FluidSolidInterface() = default;

    /**
     * Build all interface caches. The fluid FES must be scalar (vdim=1) and
     * the solid FES must be vector (vdim=Dim). The two submeshes must have
     * been created from the same parent ParMesh — we rely on that to match
     * fluid and solid GLL nodes at the interface by parent-face id.
     *
     * @param parent         parent ParMesh
     * @param fluid_sub      fluid ParSubMesh (= CreateFromDomain(parent, {fluid_attr}))
     * @param solid_sub      solid ParSubMesh (= CreateFromDomain(parent, {solid_attr}))
     * @param fluid_attr     parent-mesh element attribute of the fluid domain
     * @param solid_attr     parent-mesh element attribute of the solid domain
     * @param fluid_fes      fluid scalar FES
     * @param solid_fes      solid vector FES (vdim=Dim)
     */
    void Setup(ParMesh&               parent,
               ParSubMesh&            fluid_sub,
               ParSubMesh&            solid_sub,
               int                    fluid_attr,
               int                    solid_attr,
               ParFiniteElementSpace& fluid_fes,
               ParFiniteElementSpace& solid_fes);

    // -----------------------------------------------------------------------
    // Interface metadata
    // -----------------------------------------------------------------------

    /// Auto-detected boundary attribute shared by both submeshes at Γ_fs.
    /// Valid only after Setup(); -1 otherwise.
    int InterfaceAttribute() const { return interface_attr_; }

    /// Number of interface quadrature points local to this rank. Zero on
    /// ranks that own no interface face.
    int NumLocalQuadPoints() const { return num_local_quad_; }

    /// Per-quad outward unit normal (fluid → solid). Flattened: [i*Dim + d].
    const Vector& Normals() const { return normals_; }

    /// Per-quad face Jacobian determinant × quadrature weight. [i].
    const Vector& JacobianTimesWeight() const { return jacobian_w_; }

    /// Per-quad physical coordinates (for diagnostics + analytic tests).
    /// Flattened: [i*Dim + d].
    const Vector& QuadPositions() const { return quad_positions_; }

    /// Per-quad fluid FES scalar DOF index. [i].
    const Array<int>& FluidDofs() const { return fluid_dofs_; }

    /// Per-quad solid FES vdof index per component. Flattened: [i*Dim + d].
    const Array<int>& SolidDofs() const { return solid_dofs_; }

    /// Number of interface quadrature points this rank must exchange
    /// across MPI ranks — i.e. quads on faces whose fluid and solid
    /// elements live on different ranks. Ranks that own neither side
    /// return zero.
    int NumRemoteFluidOwnedQuadPoints() const { return num_remote_fluid_owned_quad_; }
    int NumRemoteSolidOwnedQuadPoints() const { return num_remote_solid_owned_quad_; }

    // -----------------------------------------------------------------------
    // Gather operations
    // -----------------------------------------------------------------------

    /**
     * Extract `u_s · n` at every local interface quadrature node.
     *
     * @param u_s    solid displacement ParGridFunction (vdim=Dim)
     * @param un_out resized to NumLocalQuadPoints() and filled
     */
    void ExtractSolidDisplacementNormal(const ParGridFunction& u_s,
                                        Vector&                un_out) const;

    /**
     * Extract `φ_tt` (acoustic potential second time derivative) at every
     * local interface quadrature node. Since the fluid FES is scalar and
     * the cache stores scalar DOF indices, this is a direct gather.
     */
    void ExtractFluidPotentialAccel(const ParGridFunction& phi_tt,
                                    Vector&                out) const;

    // -----------------------------------------------------------------------
    // Scatter operations (RHS contributions). Semantics:
    //   these are the collocated SEM equivalents of
    //     ∫_Γ (u_s · n) · q dΓ        (solid → fluid)
    //     ∫_Γ (-φ_tt   · n) · v_s dΓ  (fluid → solid)
    // which in MFEM would typically be expressed via
    // ParLinearForm::AddBdrFaceIntegrator. We scatter directly to the RHS
    // DOF lists built in Setup() so that every step avoids re-assembling a
    // BoundaryLFIntegrator and stays GPU-fuse-friendly (Doc §付録 K.5).
    // -----------------------------------------------------------------------

    /**
     * Add `w |J| · (u_s · n)` at every interface quadrature node to the
     * supplied fluid RHS vector at the cached fluid scalar DOFs. The
     * input vector is indexed by fluid scalar DOF (same ordering as the
     * fluid ParFiniteElementSpace), so the caller typically passes the
     * acoustic operator's `phi_tt` RHS buffer.
     */
    void ApplySolidToFluidRHS(const ParGridFunction& u_s,
                              Vector&                fluid_rhs);

    /**
     * Add `-w |J| · φ_tt · n` at every interface quadrature node to the
     * supplied solid RHS vector at the cached solid vector VDOFs. The
     * caller passes the elastic operator's acceleration RHS buffer (the
     * one that already contains `f_u − K_s u_s`).
     */
    void ApplyFluidToSolidRHS(const ParGridFunction& phi_tt,
                              Vector&                solid_rhs);

private:
    // Set by Setup:
    int  interface_attr_ = -1;
    int  num_local_quad_ =  0;

    // Non-owning references (valid while owning facade is alive).
    ParMesh*               parent_    = nullptr;
    ParSubMesh*            fluid_sub_ = nullptr;
    ParSubMesh*            solid_sub_ = nullptr;
    ParFiniteElementSpace* fluid_fes_ = nullptr;
    ParFiniteElementSpace* solid_fes_ = nullptr;
    int fluid_attr_ = 0;
    int solid_attr_ = 0;

    // Cached per-quadrature data. Layouts are documented on the accessor
    // comments above. UseDevice(true) so the forall kernels can read the
    // arrays directly on GPU.
    Vector     normals_;
    Vector     jacobian_w_;
    Vector     quad_positions_;
    Array<int> fluid_dofs_;
    Array<int> solid_dofs_;

    // Remote-face bookkeeping for interface faces whose fluid element
    // and solid element live on different MPI ranks. Each remote face
    // shows up exactly once on the fluid-owning rank and once on the
    // solid-owning rank; the rank that owns neither keeps nothing.
    // Drives the non-blocking bidirectional MPI exchange every step:
    //   - solid-owner rank packs u·n, ships it to the fluid-owner, which
    //     adds jw·(u·n) to the fluid RHS.
    //   - fluid-owner rank packs φ_tt, ships it to the solid-owner,
    //     which adds −jw·φ_tt·n to the solid RHS.
    int num_remote_fluid_owned_quad_ = 0;
    int num_remote_solid_owned_quad_ = 0;

    struct RemoteFaceRecord {
        int          peer_rank   = -1;
        int          parent_face = -1;      // rank-local parent face id (debug)
        HYPRE_BigInt canonical_key = 0;     // GLOBAL min scalar TDOF in the face —
                                            // same integer on every rank that
                                            // holds this face, used to pair
                                            // sender/receiver records.
        Array<int>   my_dofs;               // scalar fluid DOFs (fluid-owned)
                                            // OR scalar solid DOFs (solid-owned)
        Vector       normals;               // [N × Dim]
        Vector       jacobianw;             // [N]
        int          buf_offset  = 0;       // position in exchange buffer
    };
    std::vector<RemoteFaceRecord> remote_fluid_owned_;  // we own fluid side
    std::vector<RemoteFaceRecord> remote_solid_owned_;  // we own solid side

    // Per-peer exchange schedule. peer_ranks_ lists the unique peers we
    // have at least one remote face with. For each peer k:
    //   peer_fluid_owned_counts_[k] = # of OUR fluid-owned quads that
    //       talk to peer k (we send φ_tt here, we receive u·n here)
    //   peer_solid_owned_counts_[k] = # of OUR solid-owned quads that
    //       talk to peer k (we send u·n here, we receive φ_tt here)
    // Displacements are prefix sums into fluid_owned_buf_ / solid_owned_buf_.
    std::vector<int> peer_ranks_;
    std::vector<int> peer_fluid_owned_counts_;
    std::vector<int> peer_fluid_owned_displs_;
    std::vector<int> peer_solid_owned_counts_;
    std::vector<int> peer_solid_owned_displs_;

    // Reusable per-direction buffers. Each holds a scalar per interface
    // quadrature point (u·n or φ_tt).
    //   fluid_owned_buf_: size = num_remote_fluid_owned_quad_
    //   solid_owned_buf_: size = num_remote_solid_owned_quad_
    // Use pattern:
    //   ApplySolidToFluidRHS — pack u·n into solid_owned_buf_ (SEND),
    //                         receive into fluid_owned_buf_ (RECV).
    //   ApplyFluidToSolidRHS — pack φ_tt into fluid_owned_buf_ (SEND),
    //                         receive into solid_owned_buf_ (RECV).
    // Both buffers declared UseDevice(true) so the remote exchange can
    // feed device pointers directly to GPU-aware MPI_Isend/Irecv when
    // available (Read/ReadWrite), or fall back to HostRead/HostReadWrite
    // otherwise.
    Vector fluid_owned_buf_;
    Vector solid_owned_buf_;

    // --- internal helpers ---
    static int DetectInterfaceAttribute(const ParMesh&       parent,
                                        const ParSubMesh&    fluid_sub,
                                        const ParSubMesh&    solid_sub);
};

using FluidSolidInterface2D = FluidSolidInterface<2>;
using FluidSolidInterface3D = FluidSolidInterface<3>;

}  // namespace SEM

#endif  // SEM_COUPLING_FLUID_SOLID_INTERFACE_HPP
