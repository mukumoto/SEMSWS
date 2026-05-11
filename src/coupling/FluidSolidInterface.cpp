/**
 * @file FluidSolidInterface.cpp
 * @brief Fluid-solid interface gather/scatter implementation — see header for scope.
 */

#include "coupling/FluidSolidInterface.hpp"

#include <mfem.hpp>
#include "general/forall.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace SEM {

using namespace mfem;

// ---------------------------------------------------------------------------
// Interface attribute auto-detection (set-difference against parent bdr attrs)
// ---------------------------------------------------------------------------

template<int Dim>
int FluidSolidInterface<Dim>::DetectInterfaceAttribute(
    const ParMesh&    parent,
    const ParSubMesh& fluid_sub,
    const ParSubMesh& solid_sub)
{
    std::set<int> parent_bdrs;
    for (int i = 0; i < parent.bdr_attributes.Size(); ++i) {
        parent_bdrs.insert(parent.bdr_attributes[i]);
    }
    auto diff = [&](const ParSubMesh& sm) {
        std::set<int> s;
        for (int i = 0; i < sm.bdr_attributes.Size(); ++i) {
            int a = sm.bdr_attributes[i];
            if (!parent_bdrs.count(a)) s.insert(a);
        }
        return s;
    };
    const auto f_new = diff(fluid_sub);
    const auto s_new = diff(solid_sub);

    MFEM_VERIFY(f_new.size() == 1,
                "fluid submesh auto-generated bdr attrs = " << f_new.size()
                << "; expected exactly 1 (the fluid-solid interface).");
    MFEM_VERIFY(s_new.size() == 1,
                "solid submesh auto-generated bdr attrs = " << s_new.size()
                << "; expected exactly 1.");
    MFEM_VERIFY(*f_new.begin() == *s_new.begin(),
                "fluid/solid submeshes disagree on interface bdr attr ("
                << *f_new.begin() << " vs " << *s_new.begin() << ")");
    return *f_new.begin();
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

namespace {

/// Parent-entity map for a ParSubMesh, dimension-aware.
///
/// In MFEM, a boundary element of an N-dim mesh is an (N-1)-dim entity:
///   Dim=2 → boundary = edge  (1D); parent map is parent_edge_ids_
///   Dim=3 → boundary = face  (2D); parent map is parent_face_ids_
///
/// `ParSubMesh::GetParentFaceIDMap()` is only populated when the parent
/// mesh is 3D (see mfem/mesh/submesh/psubmesh.cpp: the `if (Dim == 3)`
/// guard around BuildFaceMap). In 2D it is an empty array, so
/// `parent_face_ids[face_idx]` reads past the end and segfaults.
/// Using `GetParentEdgeIDMap()` for Dim=2 gives the correct lookup.
///
/// `Mesh::GetBdrElementFaceIndex(be)` itself is dimension-uniform —
/// `be_to_face` is documented as "faces = vertices (1D), edges (2D),
/// faces (3D)" — so the returned index is automatically an edge index
/// in 2D, a face index in 3D, and the matching parent-ID array is what
/// this helper supplies.
inline const mfem::Array<int>&
GetParentBdrEntityIDMap(const ParSubMesh& sm)
{
    return (sm.Dimension() == 2) ? sm.GetParentEdgeIDMap()
                                 : sm.GetParentFaceIDMap();
}

/// Collect (submesh bdr elem index -> submesh face index) for a given attr,
/// keyed by the parent face id those bdr elems correspond to. Used so we
/// can look up "the solid bdr face that shares parent face X" when walking
/// the fluid side.
std::map<int, int>
BuildParentFaceToBdrMap(const ParSubMesh& sm, int interface_attr)
{
    std::map<int, int> map;
    const auto& parent_face_ids = GetParentBdrEntityIDMap(sm);
    for (int be = 0; be < sm.GetNBE(); ++be) {
        if (sm.GetBdrAttribute(be) != interface_attr) continue;
        const int face_idx   = sm.GetBdrElementFaceIndex(be);
        const int parent_fid = parent_face_ids[face_idx];
        map.emplace(parent_fid, face_idx);
    }
    return map;
}

}  // namespace

template<int Dim>
void FluidSolidInterface<Dim>::Setup(
    ParMesh&               parent,
    ParSubMesh&            fluid_sub,
    ParSubMesh&            solid_sub,
    int                    fluid_attr,
    int                    solid_attr,
    ParFiniteElementSpace& fluid_fes,
    ParFiniteElementSpace& solid_fes)
{
    parent_    = &parent;
    fluid_sub_ = &fluid_sub;
    solid_sub_ = &solid_sub;
    fluid_fes_ = &fluid_fes;
    solid_fes_ = &solid_fes;
    fluid_attr_ = fluid_attr;
    solid_attr_ = solid_attr;

    MFEM_VERIFY(fluid_fes.GetVDim() == 1,
                "fluid FES must be scalar (vdim=1), got " << fluid_fes.GetVDim());
    MFEM_VERIFY(solid_fes.GetVDim() == Dim,
                "solid FES must be vector with vdim=" << Dim << ", got "
                << solid_fes.GetVDim());

    interface_attr_ = DetectInterfaceAttribute(parent, fluid_sub, solid_sub);

    // Walk the fluid-side interface bdr faces and, for each,
    // look up the matching solid-side face via the parent face id.
    const auto solid_by_parent_face =
        BuildParentFaceToBdrMap(solid_sub, interface_attr_);

    // Dimension-aware parent-entity map (see GetParentBdrEntityIDMap).
    const auto& fluid_pf = GetParentBdrEntityIDMap(fluid_sub);

    // Build a scalar "coordinate grid function" on the fluid FES so we can
    // read off the physical xyz of each fluid DOF (used for Quad positions
    // cache + analytic-field tests). ProjectCoefficient once is cheaper
    // than calling the element transformation per DOF.
    ParGridFunction fluid_pos_gf(&fluid_fes, (real_t*)nullptr);
    ParFiniteElementSpace fluid_vec_fes(&fluid_sub, fluid_fes.FEColl(), Dim,
                                        mfem::Ordering::byNODES);
    ParGridFunction fluid_nodes(&fluid_vec_fes);
    VectorFunctionCoefficient id_coef(Dim, [](const Vector& x, Vector& v) {
        v = x;
    });
    fluid_nodes.ProjectCoefficient(id_coef);
    const int fluid_nscalar = fluid_fes.GetNDofs();

    // Build fluid/solid submesh DOF → parent DOF maps using MFEM's own
    // SubMeshUtils::BuildVdofToVdofMap (the internal routine ParTransferMap
    // uses). Crucially, we feed in the ACTUAL FES objects (fluid_fes is
    // vdim=1 scalar H1, solid_fes is vdim=Dim vector H1) and match each
    // side with a parent FES of the SAME vdim. Mixing vdims (e.g. using a
    // freshly-built vdim=1 FES on solid_sub alongside solid_fes at vdim=Dim
    // for DofToVDof) is not safe: MFEM does not promise the scalar-DOF
    // numbering of two separately-constructed ParFESes with different
    // vdim is identical, so the cross-lookup can land on the wrong VDOF.
    //
    // Reuse fluid_fes's FEColl for parent: fluid/solid/parent share the
    // same polynomial order on this H1 space, so the FECollection carries
    // only the order + continuity/space type and is interchangeable.
    ParFiniteElementSpace parent_scalar_fes(&parent, fluid_fes.FEColl(), 1,
                                            mfem::Ordering::byNODES);
    ParFiniteElementSpace parent_vec_fes(&parent, fluid_fes.FEColl(), Dim,
                                         mfem::Ordering::byNODES);

    mfem::Array<int> fluid_to_parent_scalar;
    mfem::SubMeshUtils::BuildVdofToVdofMap(
        fluid_fes, parent_scalar_fes,
        mfem::SubMesh::From::Domain,
        fluid_sub.GetParentElementIDMap(),
        fluid_to_parent_scalar);

    mfem::Array<int> solid_to_parent_vdof;
    mfem::SubMeshUtils::BuildVdofToVdofMap(
        solid_fes, parent_vec_fes,
        mfem::SubMesh::From::Domain,
        solid_sub.GetParentElementIDMap(),
        solid_to_parent_vdof);

    // Invert solid_to_parent (VDOF level) so we can look up
    // parent_vdof → solid_vdof directly.
    std::map<int, int> parent_vdof_to_solid_vdof;
    for (int i = 0; i < solid_to_parent_vdof.Size(); ++i) {
        const int p = mfem::FiniteElementSpace::DecodeDof(
            solid_to_parent_vdof[i]);
        parent_vdof_to_solid_vdof[p] = i;
    }
    // parent_scalar_fes and parent_vec_fes share the same scalar DOF
    // numbering for the same (mesh, FEColl). That's what makes the
    // parent_vdof = parent_scalar + d * parent_nscalar arithmetic below
    // valid (one-shot verified while developing; the identity-projection
    // cross-check now lives in coupling_probe_interface).
    const int parent_nscalar = parent_scalar_fes.GetNDofs();

    // Per-submesh scalar-DOF → GLOBAL SCALAR TDOF maps on the parent FES.
    // We use these in the remote-face path to impose a GLOBALLY CANONICAL
    // ordering of a face's quadrature points (sort by global TDOF number)
    // so the solid-owner's send buffer and the fluid-owner's recv buffer
    // lay quads out in the same order — independent of any per-rank
    // submesh LEX ordering quirks or per-rank parent-DOF renumbering.
    //
    // Important: parent_scalar_fes numbers LDOFs rank-locally, so the raw
    // "parent_scalar_dof" L-index is NOT comparable across ranks. The
    // GLOBAL SCALAR TDOF (HYPRE_BigInt) IS comparable — it's the HYPRE
    // global row index shared by every rank that holds the DOF locally
    // or as a ghost. GetGlobalScalarTDofNumber works on any LDOF,
    // including ghosts, returning the same number on every rank.
    const int solid_nscalar_global_map = solid_fes.GetNDofs();
    std::vector<HYPRE_BigInt> fluid_submesh_to_parent_gtdof(fluid_fes.GetNDofs());
    std::vector<HYPRE_BigInt> solid_submesh_to_parent_gtdof(solid_nscalar_global_map);
    for (int k = 0; k < fluid_fes.GetNDofs(); ++k) {
        const int parent_ldof = mfem::FiniteElementSpace::DecodeDof(
            fluid_to_parent_scalar[k]);
        fluid_submesh_to_parent_gtdof[k] =
            parent_scalar_fes.GetGlobalScalarTDofNumber(parent_ldof);
    }
    for (int k = 0; k < solid_nscalar_global_map; ++k) {
        const int parent_vd   = mfem::FiniteElementSpace::DecodeDof(
            solid_to_parent_vdof[k]);
        const int parent_ldof = parent_vd % parent_nscalar;
        solid_submesh_to_parent_gtdof[k] =
            parent_scalar_fes.GetGlobalScalarTDofNumber(parent_ldof);
    }

    // Temporary vectors — copy into the final flat storage at the end so the
    // cache layout stays packed.
    std::vector<real_t> tmp_normals, tmp_jacw, tmp_pos;
    std::vector<int>    tmp_fluid_dofs, tmp_solid_dofs;

    // Face-local GLL integration rule, tensor product of 1D GLL. Points
    // coincide geometrically with face H1 Lagrange DOFs — we do NOT rely
    // on their enumeration order matching; each DOF is paired with the
    // IR point at the same physical xyz below (nearest-neighbour lookup).
    //
    // NOTE: derive order from the FE collection, not `fes.GetOrder(0)`,
    // so that ranks with zero local fluid elements (the MPI-spanning
    // interface case handled via the remote-MPI path below) don't crash here.
    const auto* fec = dynamic_cast<const H1_FECollection*>(fluid_fes.FEColl());
    MFEM_VERIFY(fec, "fluid FES must use H1_FECollection");
    const int order = fec->GetOrder();
    IntegrationRules gll_rules(0, Quadrature1D::GaussLobatto);

    // Pre-compute the permutation from GLL-IR lex order to MFEM's face-
    // internal DOF ordering, once per Setup. H1_*Element::GetLexicographicOrdering()
    // is MFEM's official way to get this for any polynomial order + geometry.
    // For a 3D hex face the face FE is H1_QuadrilateralElement; for a 2D
    // segment face it's H1_SegmentElement. The returned `lex_perm` has the
    // property: lex_perm[lex_idx] = MFEM face-internal DOF index for the
    // lex_idx-th tensor-product GLL point. Combined with GetFaceDofs (which
    // returns DOFs in the same MFEM face-internal ordering), we can pair
    // IR point q directly with fluid face DOF f_dofs[lex_perm[q]] — no
    // physical-position nearest-neighbour search needed.
    std::vector<int> lex_perm;
    {
        const auto btype = fec->GetBasisType();
        if constexpr (Dim == 3) {
            H1_QuadrilateralElement face_fe(order, btype);
            const Array<int>& a = face_fe.GetLexicographicOrdering();
            lex_perm.assign(a.begin(), a.end());
        } else {
            H1_SegmentElement face_fe(order, btype);
            const Array<int>& a = face_fe.GetLexicographicOrdering();
            lex_perm.assign(a.begin(), a.end());
        }
    }

    for (int be = 0; be < fluid_sub.GetNBE(); ++be) {
        if (fluid_sub.GetBdrAttribute(be) != interface_attr_) continue;
        const int fluid_face_idx = fluid_sub.GetBdrElementFaceIndex(be);
        const int parent_face    = fluid_pf[fluid_face_idx];

        auto it = solid_by_parent_face.find(parent_face);
        if (it == solid_by_parent_face.end()) {
            // No matching solid face on this rank — the interface face
            // straddles an MPI partition boundary — handled by the remote
            // exchange path; no per-face action needed here.
            continue;
        }
        const int solid_face_idx = it->second;

        // Fluid side: scalar DOFs on the face (one DOF per GLL node).
        Array<int> f_dofs;
        fluid_fes.GetFaceDofs(fluid_face_idx, f_dofs);

        // Solid side: vdim=Dim DOFs on the face. GetFaceDofs returns the
        // scalar-space DOFs; we expand to vdofs per component below.
        Array<int> s_scalar_dofs;
        solid_fes.GetFaceDofs(solid_face_idx, s_scalar_dofs);
        MFEM_VERIFY(f_dofs.Size() == s_scalar_dofs.Size(),
                    "fluid/solid face DOF count mismatch at parent face "
                    << parent_face << " (fluid=" << f_dofs.Size()
                    << ", solid=" << s_scalar_dofs.Size() << "). Likely a "
                    "polynomial-order / face-orientation issue.");

        // Every geometric quantity at the interface is evaluated
        // **per quadrature point** — we never fall back to "assume the
        // face is planar, reuse face-centre values". For Cartesian meshes
        // the per-quad values collapse to one shared value anyway; for
        // curved or non-convex interfaces the per-quad evaluation is the
        // only correct choice, and future non-matching / mortar coupling
        // will need it too.
        auto* ftr = fluid_sub.GetFaceElementTransformations(fluid_face_idx);
        MFEM_VERIFY(ftr, "fluid submesh FaceElementTransformations null at "
                    << fluid_face_idx);

        // Fluid-interior element centroid — the only FIXED reference used
        // in the per-quad sign-flip test below. Computed once per face
        // because it's a property of the element, not of the quad.
        Vector fluid_centroid(Dim);
        {
            const int fluid_sub_elem = ftr->Elem1No;
            ElementTransformation* etr =
                fluid_sub.GetElementTransformation(fluid_sub_elem);
            IntegrationPoint ec;
            ec.Init(0);
            if (Dim == 2) { ec.x = 0.5; ec.y = 0.5; }
            else          { ec.x = 0.5; ec.y = 0.5; ec.z = 0.5; }
            etr->SetIntPoint(&ec);
            etr->Transform(ec, fluid_centroid);
        }

        // Evaluate the face GLL integration rule once: for each IR point
        // record physical xyz, the area-weight w·|J|, and a UNIT normal
        // oriented fluid→solid. Storing per-quad normals avoids the old
        // "one normal per face" shortcut — safe for curved faces and any
        // future non-conforming interfaces.
        const Geometry::Type fgeom = ftr->GetGeometryType();
        const IntegrationRule& ir  = gll_rules.Get(fgeom, 2 * order - 1);
        const int nq_ir = ir.GetNPoints();
        MFEM_VERIFY(nq_ir == f_dofs.Size(),
                    "face GLL IR has " << nq_ir << " points but face DOF "
                    "count = " << f_dofs.Size()
                    << " — polynomial order mismatch between H1 space and "
                    "chosen GLL rule.");
        std::vector<real_t> ir_x(nq_ir * Dim), ir_jw(nq_ir);
        std::vector<real_t> ir_n(nq_ir * Dim);
        for (int q = 0; q < nq_ir; ++q) {
            const IntegrationPoint& ip = ir.IntPoint(q);
            ftr->SetAllIntPoints(&ip);

            // Physical position + area weight.
            Vector pos(Dim);
            ftr->Transform(ip, pos);
            for (int d = 0; d < Dim; ++d) ir_x[q * Dim + d] = pos[d];
            ir_jw[q] = ip.weight * ftr->Weight();

            // Per-quad normal: CalcOrtho returns a vector with |n| = |J|
            // aligned with the face's reference-space "outward" orientation.
            // Normalise, then flip sign so that the normal points
            // fluid→solid using **this quad's** offset from the fluid
            // element centroid (no face-level orientation assumption).
            Vector n_q(Dim);
            CalcOrtho(ftr->Jacobian(), n_q);
            const real_t nrm = n_q.Norml2();
            MFEM_VERIFY(nrm > 0,
                        "zero-length face normal at interface quadrature");
            n_q /= nrm;
            real_t align = 0.0;
            for (int d = 0; d < Dim; ++d) {
                align += (pos[d] - fluid_centroid[d]) * n_q[d];
            }
            if (align < 0) n_q.Neg();
            for (int d = 0; d < Dim; ++d) ir_n[q * Dim + d] = n_q[d];
        }

        // Iterate in LEX (= GLL IR) order. For each IR point lex_idx:
        //   - f_dofs[ lex_perm[lex_idx] ]  → the global fluid scalar DOF
        //   - ir_jw[ lex_idx ]             → its Jacobian-weight (direct index)
        //
        // Scalar/vector FES handling: fluid is vdim=1 (scalar), solid is
        // vdim=Dim (vector). We keep the two maps at their NATURAL vdims
        // (fluid_to_parent_scalar, solid_to_parent_vdof) and bridge via the
        // byNODES parent VDOF arithmetic parent_vdof_d = parent_scalar
        // + d * parent_nscalar. That avoids any cross-vdim lookup that
        // assumes identical scalar numbering between two freshly-built
        // parallel FES instances with different vdim.
        const real_t* fn = fluid_nodes.HostRead();
        for (int lex_idx = 0; lex_idx < nq_ir; ++lex_idx) {
            const int mfem_face_local = lex_perm[lex_idx];
            const int fluid_dof_idx   = f_dofs[mfem_face_local];
            tmp_fluid_dofs.push_back(fluid_dof_idx);

            const int parent_scalar_dof = mfem::FiniteElementSpace::DecodeDof(
                fluid_to_parent_scalar[fluid_dof_idx]);
            for (int d = 0; d < Dim; ++d) {
                const int parent_vdof = parent_scalar_dof + d * parent_nscalar;
                auto it = parent_vdof_to_solid_vdof.find(parent_vdof);
                MFEM_VERIFY(it != parent_vdof_to_solid_vdof.end(),
                            "fluid DOF " << fluid_dof_idx << " (parent scalar "
                            << parent_scalar_dof << ", component " << d
                            << " → parent vdof " << parent_vdof << ") has no "
                            "matching solid VDOF. Non-conforming meshes?");
                tmp_solid_dofs.push_back(it->second);
            }

            // Per-quad UNIT normal (fluid→solid), evaluated at THIS quad
            // via CalcOrtho + alignment check — no planar-face shortcut.
            for (int d = 0; d < Dim; ++d) {
                tmp_normals.push_back(ir_n[lex_idx * Dim + d]);
            }

            // Direct IR index — jws[lex_idx] is the weight at the IR point
            // that LEX-corresponds to this face DOF.
            tmp_jacw.push_back(ir_jw[lex_idx]);

            // Quad position: physical coord of this fluid DOF.
            for (int d = 0; d < Dim; ++d) {
                tmp_pos.push_back(fn[fluid_dof_idx + d * fluid_nscalar]);
            }
        }
    }

    num_local_quad_ = (int)tmp_fluid_dofs.size();

    // -----------------------------------------------------------------
    // Remote: build per-rank metadata for interface faces whose fluid
    // and solid elements live on DIFFERENT MPI ranks. We harvest peer
    // rank information from the PARENT mesh (only consistent view across
    // both domains), then re-walk each submesh's boundary to pull the
    // local-side DOF / normal / jacobianw data for remote faces. The
    // final result is two vectors of RemoteFaceRecord plus a per-peer
    // exchange schedule ready for non-blocking MPI.
    // -----------------------------------------------------------------
    parent.ExchangeFaceNbrData();  // idempotent if already called

    // Map parent face id -> peer rank for remote interface faces that
    // touch THIS rank (as either fluid or solid owner).
    std::map<int, int> peer_rank_by_parent_face;
    for (int pf = 0; pf < parent.GetNumFaces(); ++pf) {
        const auto info = parent.GetFaceInformation(pf);
        if (!info.IsInterior()) continue;
        if (info.element[0].location != Mesh::ElementLocation::Local) continue;
        if (info.element[1].location != Mesh::ElementLocation::FaceNbr) continue;

        const int local_elem_idx  = info.element[0].index;
        const int remote_elem_idx = info.element[1].index;
        const int local_attr  = parent.GetAttribute(local_elem_idx);
        MFEM_VERIFY(remote_elem_idx >= 0 &&
                    remote_elem_idx < parent.face_nbr_elements.Size(),
                    "face-nbr element index out of range");
        const int remote_attr =
            parent.face_nbr_elements[remote_elem_idx]->GetAttribute();
        if (local_attr == remote_attr) continue;
        const bool is_interface =
            (local_attr == fluid_attr && remote_attr == solid_attr) ||
            (local_attr == solid_attr && remote_attr == fluid_attr);
        if (!is_interface) continue;

        int group = -1;
        for (int fn = 0; fn < parent.GetNFaceNeighbors(); ++fn) {
            if (remote_elem_idx >= parent.face_nbr_elements_offset[fn] &&
                remote_elem_idx <  parent.face_nbr_elements_offset[fn + 1]) {
                group = fn; break;
            }
        }
        MFEM_VERIFY(group >= 0, "could not resolve face-nbr group");
        peer_rank_by_parent_face[pf] = parent.GetFaceNbrRank(group);
    }

    // Helper: for a submesh boundary face we know is part of the interface,
    // compute the per-quadrature geometric data (normal, jacobianw) and
    // the face DOF list on that submesh side, paired by nearest-physical
    // position so the flat arrays are consistent across sides.
    //
    // Canonical quadrature ordering: the fluid-owner and the solid-owner of
    // a remote face each build its own FaceElementTransformations from its
    // own submesh view. Those two transformations can enumerate LEX points
    // in different orders (opposite orientations of the same parent face).
    // To guarantee send/recv pair up point-for-point, every side sorts the
    // face's quads by parent_scalar_dof (a globally unique integer key)
    // before emitting into the flat record arrays. Caller supplies the
    // submesh→parent-scalar map for the side it's walking.
    auto build_face_cache =
        [&](ParSubMesh& sm, int face_idx, const ParGridFunction& nodes_gf,
            int nscalar_of_nodes,
            bool flip_normal_for_fluid,
            const std::vector<HYPRE_BigInt>& submesh_to_parent_gtdof,
            std::vector<int>& out_dofs_per_quad_scalar,
            std::vector<real_t>& out_normals,
            std::vector<real_t>& out_jw,
            HYPRE_BigInt* out_min_gtdof /*nullable*/)
        -> void
    {
        auto* ftr = sm.GetFaceElementTransformations(face_idx);
        MFEM_VERIFY(ftr, "submesh face transformation null");

        // Submesh-interior element centroid — the only fixed reference
        // for the per-quad sign-flip test. Everything else is per-quad.
        Vector adj_centroid(Dim);
        {
            const int adj_elem = ftr->Elem1No;
            ElementTransformation* etr =
                sm.GetElementTransformation(adj_elem);
            IntegrationPoint ec;
            ec.Init(0);
            if (Dim == 2) { ec.x = 0.5; ec.y = 0.5; }
            else          { ec.x = 0.5; ec.y = 0.5; ec.z = 0.5; }
            etr->SetIntPoint(&ec);
            etr->Transform(ec, adj_centroid);
        }
        // When called on the SOLID submesh, `flip_normal_for_fluid`
        // inverts the outward-from-submesh direction so the stored
        // normal still points fluid→solid. Applied per quad below.
        const real_t orient_sign = flip_normal_for_fluid ? -1.0 : 1.0;

        // Face DOFs + IR data (same code path as local face handling).
        Array<int> face_dofs;
        // Callers must pass fluid_fes or solid_fes (scalar form) as a
        // closure — we recover the right FES from nodes_gf's space.
        const ParFiniteElementSpace* fes_ptr = nodes_gf.ParFESpace();
        const_cast<ParFiniteElementSpace*>(fes_ptr)->GetFaceDofs(face_idx, face_dofs);

        const Geometry::Type fgeom = ftr->GetGeometryType();
        const IntegrationRule& ir = gll_rules.Get(fgeom, 2 * order - 1);
        const int nq = ir.GetNPoints();
        MFEM_VERIFY(nq == face_dofs.Size(),
                    "face IR points (" << nq << ") != face DOFs ("
                    << face_dofs.Size() << ")");
        std::vector<real_t> ir_x(nq * Dim), ir_jw(nq);
        std::vector<real_t> ir_n(nq * Dim);   // per-quad unit normal (fluid→solid)
        for (int q = 0; q < nq; ++q) {
            const IntegrationPoint& ip = ir.IntPoint(q);
            ftr->SetAllIntPoints(&ip);

            Vector pos(Dim);
            ftr->Transform(ip, pos);
            for (int d = 0; d < Dim; ++d) ir_x[q * Dim + d] = pos[d];
            ir_jw[q] = ip.weight * ftr->Weight();

            Vector n_q(Dim);
            CalcOrtho(ftr->Jacobian(), n_q);
            const real_t nrm = n_q.Norml2();
            MFEM_VERIFY(nrm > 0,
                        "zero-length face normal at interface quadrature");
            n_q /= nrm;
            real_t align = 0.0;
            for (int d = 0; d < Dim; ++d) {
                align += (pos[d] - adj_centroid[d]) * n_q[d];
            }
            if (align < 0) n_q.Neg();
            // Reorient solid-side normals to the fluid→solid convention.
            n_q *= orient_sign;
            for (int d = 0; d < Dim; ++d) ir_n[q * Dim + d] = n_q[d];
        }

        out_dofs_per_quad_scalar.reserve(
            out_dofs_per_quad_scalar.size() + nq);
        out_normals.reserve(out_normals.size() + nq * Dim);
        out_jw.reserve(out_jw.size() + nq);

        const real_t* np_data = nodes_gf.HostRead();
        (void)np_data;  // no longer needed for pairing, only for position queries
        // Enumerate quads in LEX order, attach each to its submesh scalar
        // DOF, parent scalar DOF, jw and unit normal. Sort by
        // parent_scalar_dof (global TDOF) so the fluid-owner and solid-
        // owner produce identical per-face ordering — the normals and
        // weights are carried along so the post-sort output still pairs
        // each quad with its own geometry (not a face-averaged one).
        struct QEntry {
            HYPRE_BigInt gtdof;        // global scalar TDOF, cross-rank unique
            int          submesh_dof;
            real_t       jw;
            real_t       n[Dim];       // per-quad unit normal (fluid→solid)
        };
        std::vector<QEntry> entries;
        entries.reserve(nq);
        for (int lex_idx = 0; lex_idx < nq; ++lex_idx) {
            const int mfem_face_local = lex_perm[lex_idx];
            const int dof_idx         = face_dofs[mfem_face_local];
            MFEM_VERIFY(dof_idx >= 0 &&
                        dof_idx < (int)submesh_to_parent_gtdof.size(),
                        "face scalar dof out of submesh range");
            const HYPRE_BigInt gt = submesh_to_parent_gtdof[dof_idx];
            QEntry e;
            e.gtdof       = gt;
            e.submesh_dof = dof_idx;
            e.jw          = ir_jw[lex_idx];
            for (int d = 0; d < Dim; ++d) e.n[d] = ir_n[lex_idx * Dim + d];
            entries.push_back(e);
        }
        std::sort(entries.begin(), entries.end(),
                  [](const QEntry& a, const QEntry& b) {
                      return a.gtdof < b.gtdof;
                  });
        if (out_min_gtdof) {
            *out_min_gtdof = entries.empty()
                ? std::numeric_limits<HYPRE_BigInt>::max()
                : entries.front().gtdof;
        }
        for (const auto& e : entries) {
            out_dofs_per_quad_scalar.push_back(e.submesh_dof);
            for (int d = 0; d < Dim; ++d) out_normals.push_back(e.n[d]);
            out_jw.push_back(e.jw);
        }
    };

    // Identity-projection coord GF for solid_sub (analogous to fluid_nodes).
    ParFiniteElementSpace solid_vec_fes(&solid_sub, solid_fes.FEColl(), Dim,
                                        mfem::Ordering::byNODES);
    ParGridFunction solid_nodes(&solid_vec_fes);
    solid_nodes.ProjectCoefficient(id_coef);
    const int solid_nscalar = solid_fes.GetNDofs();

    // (The one-shot physical-position sanity check that used to live
    // here has been retired — the parent-DOF routing is exact by
    // construction via SubMeshUtils::BuildVdofToVdofMap. If future bugs
    // reintroduce cross-side mismatches, coupling_probe_interface exposes
    // the face-area and u·n / φ̈ integration checks that reveal them.)

    // PASS: re-walk fluid_sub bdrs to pick up the remote-fluid-owned faces
    // that were skipped in the local-face loop (no matching solid bdr on
    // this rank == MPI-spanning).
    for (int be = 0; be < fluid_sub.GetNBE(); ++be) {
        if (fluid_sub.GetBdrAttribute(be) != interface_attr_) continue;
        const int face_idx = fluid_sub.GetBdrElementFaceIndex(be);
        const int pf = fluid_pf[face_idx];
        if (solid_by_parent_face.count(pf)) continue;   // handled as local

        auto it = peer_rank_by_parent_face.find(pf);
        if (it == peer_rank_by_parent_face.end()) {
            // If we get here the parent mesh's topology disagrees with
            // ParSubMesh: ParSubMesh sees this edge/face as an interface
            // (fluid on one side, solid on other) but the parent's
            // GetFaceInformation claims it has no second element.
            //
            // The most common cause in practice is a mesh built by Cubit
            // with `subtract … keep_tool` WITHOUT an explicit
            // `merge vertex all; merge curve all` BEFORE the surface
            // merge: the subtract leaves duplicate vertices/curves along
            // the fluid-solid interface, so fluid and solid elements are
            // geometrically adjacent but topologically disconnected.
            // See examples/fun/semsws_coupled_2d/generate_jou.py for
            // the correct merge sequence.
            MFEM_ABORT(
                "FluidSolidInterface::Setup: parent face " << pf << " is "
                "tagged as fluid-solid interface by ParSubMesh but is "
                "classified as boundary (no second element) in the parent "
                "ParMesh. Likely mesh bug: fluid and solid elements are "
                "not topologically connected at the interface. If the mesh "
                "came from Cubit, add `merge vertex all; merge curve all` "
                "BEFORE `merge surface all`.");
        }
        std::vector<int>    rec_dofs;
        std::vector<real_t> rec_normals, rec_jw;
        HYPRE_BigInt        rec_min_gtdof = 0;
        build_face_cache(fluid_sub, face_idx, fluid_nodes, fluid_nscalar,
                         /*flip_normal_for_fluid=*/false,
                         fluid_submesh_to_parent_gtdof,
                         rec_dofs, rec_normals, rec_jw, &rec_min_gtdof);
        RemoteFaceRecord rec;
        rec.peer_rank     = it->second;
        rec.parent_face   = pf;
        rec.canonical_key = rec_min_gtdof;
        const int N = (int)rec_dofs.size();
        rec.my_dofs.SetSize(N);
        rec.normals.SetSize(N * Dim);
        rec.jacobianw.SetSize(N);
        for (int i = 0; i < N; ++i) {
            rec.my_dofs[i]   = rec_dofs[i];
            rec.jacobianw[i] = rec_jw[i];
            for (int d = 0; d < Dim; ++d) {
                rec.normals[i * Dim + d] = rec_normals[i * Dim + d];
            }
        }
        rec.normals.UseDevice(true);
        rec.jacobianw.UseDevice(true);
        num_remote_fluid_owned_quad_ += N;
        remote_fluid_owned_.push_back(std::move(rec));
    }

    // PASS: solid_sub bdrs, remote-solid-owned only.
    // Build a quick "parent face -> fluid bdr" map so we can distinguish
    // local faces (handled in the fluid walk) from remote-solid-owned.
    std::map<int, int> fluid_by_parent_face =
        BuildParentFaceToBdrMap(fluid_sub, interface_attr_);
    for (int be = 0; be < solid_sub.GetNBE(); ++be) {
        if (solid_sub.GetBdrAttribute(be) != interface_attr_) continue;
        const int face_idx = solid_sub.GetBdrElementFaceIndex(be);
        const Array<int>& solid_pf = GetParentBdrEntityIDMap(solid_sub);
        const int pf = solid_pf[face_idx];
        if (fluid_by_parent_face.count(pf)) continue;   // local

        auto it = peer_rank_by_parent_face.find(pf);
        if (it == peer_rank_by_parent_face.end()) {
            MFEM_ABORT(
                "FluidSolidInterface::Setup: parent face " << pf << " is "
                "tagged as fluid-solid interface by ParSubMesh but is "
                "classified as boundary (no second element) in the parent "
                "ParMesh. Likely mesh bug: fluid and solid elements are "
                "not topologically connected at the interface. If the mesh "
                "came from Cubit, add `merge vertex all; merge curve all` "
                "BEFORE `merge surface all`.");
        }
        std::vector<int>    rec_dofs;
        std::vector<real_t> rec_normals, rec_jw;
        HYPRE_BigInt        rec_min_gtdof = 0;
        build_face_cache(solid_sub, face_idx, solid_nodes, solid_nscalar,
                         /*flip_normal_for_fluid=*/true,
                         solid_submesh_to_parent_gtdof,
                         rec_dofs, rec_normals, rec_jw, &rec_min_gtdof);
        RemoteFaceRecord rec;
        rec.peer_rank     = it->second;
        rec.parent_face   = pf;
        rec.canonical_key = rec_min_gtdof;
        const int N = (int)rec_dofs.size();
        rec.my_dofs.SetSize(N);
        rec.normals.SetSize(N * Dim);
        rec.jacobianw.SetSize(N);
        for (int i = 0; i < N; ++i) {
            rec.my_dofs[i]   = rec_dofs[i];
            rec.jacobianw[i] = rec_jw[i];
            for (int d = 0; d < Dim; ++d) {
                rec.normals[i * Dim + d] = rec_normals[i * Dim + d];
            }
        }
        rec.normals.UseDevice(true);
        rec.jacobianw.UseDevice(true);
        num_remote_solid_owned_quad_ += N;
        remote_solid_owned_.push_back(std::move(rec));
    }

    // Build the per-peer exchange schedule. Records are sorted by peer
    // rank so each peer's quadrature contributions land in a contiguous
    // buffer slice. Fluid-owned and solid-owned are tracked independently
    // because the two Apply* calls swap send/recv roles between them.
    auto assign_peer_offsets_and_counts =
        [&](std::vector<RemoteFaceRecord>& list,
            std::map<int, int>& counts) {
        // Sort primarily by peer_rank so each peer's records are contiguous
        // (the MPI exchange is per-peer and uses displs into a packed buffer),
        // and SECONDARILY by canonical_key (the face's minimum global scalar
        // TDOF number) so that the sender side and the receiver side produce
        // the SAME ordering for the faces they exchange. canonical_key is
        // globally unique and stable across ranks, so both ranks sort to the
        // identical order without any extra communication. Without this
        // secondary key, std::sort is free to reorder tied records, which
        // at np≥4 (where one rank has multiple faces going to the same peer)
        // scrambles the quadrature-point pairing across ranks — u·n sent
        // from face F_a lands on the receiver's fluid DOFs for face F_c.
        std::sort(list.begin(), list.end(),
                  [](const RemoteFaceRecord& a, const RemoteFaceRecord& b) {
                      if (a.peer_rank != b.peer_rank) return a.peer_rank < b.peer_rank;
                      return a.canonical_key < b.canonical_key;
                  });
        int offset = 0;
        for (auto& r : list) {
            r.buf_offset = offset;
            counts[r.peer_rank] += r.my_dofs.Size();
            offset += r.my_dofs.Size();
        }
    };
    std::map<int, int> fluid_owned_per_peer;
    std::map<int, int> solid_owned_per_peer;
    assign_peer_offsets_and_counts(remote_fluid_owned_, fluid_owned_per_peer);
    assign_peer_offsets_and_counts(remote_solid_owned_, solid_owned_per_peer);

    std::set<int> all_peers;
    for (auto& kv : fluid_owned_per_peer) all_peers.insert(kv.first);
    for (auto& kv : solid_owned_per_peer) all_peers.insert(kv.first);
    peer_ranks_.assign(all_peers.begin(), all_peers.end());

    const int npeers = (int)peer_ranks_.size();
    peer_fluid_owned_counts_.assign(npeers, 0);
    peer_fluid_owned_displs_.assign(npeers, 0);
    peer_solid_owned_counts_.assign(npeers, 0);
    peer_solid_owned_displs_.assign(npeers, 0);
    int f_displ = 0, s_displ = 0;
    for (int k = 0; k < npeers; ++k) {
        const int p = peer_ranks_[k];
        peer_fluid_owned_counts_[k] = fluid_owned_per_peer[p];
        peer_solid_owned_counts_[k] = solid_owned_per_peer[p];
        peer_fluid_owned_displs_[k] = f_displ;
        peer_solid_owned_displs_[k] = s_displ;
        f_displ += peer_fluid_owned_counts_[k];
        s_displ += peer_solid_owned_counts_[k];
    }
    fluid_owned_buf_.SetSize(std::max(f_displ, 1));
    solid_owned_buf_.SetSize(std::max(s_displ, 1));
    fluid_owned_buf_.UseDevice(true);
    solid_owned_buf_.UseDevice(true);

    normals_.SetSize(num_local_quad_ * Dim);
    jacobian_w_.SetSize(num_local_quad_);
    quad_positions_.SetSize(num_local_quad_ * Dim);
    fluid_dofs_.SetSize(num_local_quad_);
    solid_dofs_.SetSize(num_local_quad_ * Dim);

    for (int i = 0; i < num_local_quad_; ++i) {
        fluid_dofs_[i] = tmp_fluid_dofs[i];
        jacobian_w_[i] = tmp_jacw[i];
        for (int d = 0; d < Dim; ++d) {
            normals_[i * Dim + d]        = tmp_normals[i * Dim + d];
            quad_positions_[i * Dim + d] = tmp_pos[i * Dim + d];
            solid_dofs_[i * Dim + d]     = tmp_solid_dofs[i * Dim + d];
        }
    }

    // Make cached Vector arrays device-resident so
    // Apply*/Extract* forall kernels operate on device pointers.
    // mfem::Array handles device residency implicitly through its
    // Read()/Write() accessors (no UseDevice flag). On CPU-only MFEM
    // builds these are no-op pass-throughs.
    normals_.UseDevice(true);
    jacobian_w_.UseDevice(true);
    // One-shot host→device copies so the first forall doesn't pay the
    // transfer cost.
    (void)normals_.Read();
    (void)jacobian_w_.Read();
    (void)fluid_dofs_.Read();
    (void)solid_dofs_.Read();
}

// ---------------------------------------------------------------------------
// Gather operations
// ---------------------------------------------------------------------------

template<int Dim>
void FluidSolidInterface<Dim>::ExtractSolidDisplacementNormal(
    const ParGridFunction& u_s, Vector& un_out) const
{
    un_out.SetSize(num_local_quad_);
    un_out.UseDevice(true);
    if (num_local_quad_ == 0) return;

    // Gather on device. Pure read-gather-write pattern, no
    // scatter involved — no atomic needed, GPU-safe as-is.
    const int   N       = num_local_quad_;
    constexpr int D     = Dim;
    const auto  u_data  = u_s.Read();
    const auto  sdofs   = solid_dofs_.Read();
    const auto  nrms    = normals_.Read();
    auto        out     = un_out.Write();
    mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
        real_t un = 0.0;
        for (int d = 0; d < D; ++d) {
            un += u_data[sdofs[i * D + d]] * nrms[i * D + d];
        }
        out[i] = un;
    });
}

template<int Dim>
void FluidSolidInterface<Dim>::ExtractFluidPotentialAccel(
    const ParGridFunction& phi_tt, Vector& out) const
{
    out.SetSize(num_local_quad_);
    out.UseDevice(true);
    if (num_local_quad_ == 0) return;

    const int  N       = num_local_quad_;
    const auto p_data  = phi_tt.Read();
    const auto fdofs   = fluid_dofs_.Read();
    auto       o_data  = out.Write();
    mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
        o_data[i] = p_data[fdofs[i]];
    });
}

// ---------------------------------------------------------------------------
// Scatter (RHS contributions)
// ---------------------------------------------------------------------------

namespace {

constexpr int kTagUn     = 0xFC01;   // solid → fluid exchange
constexpr int kTagPhiTt  = 0xFC02;   // fluid → solid exchange

inline MPI_Datatype MPIRealT()
{
    return (sizeof(real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE;
}

}  // namespace

template<int Dim>
void FluidSolidInterface<Dim>::ApplySolidToFluidRHS(
    const ParGridFunction& u_s, Vector& fluid_rhs)
{
    // -----------------------------------------------------------------
    // Local contributions: gather u·n on device and scatter into the
    // fluid RHS at cached fluid DOFs.
    //
    // Adjacent interface faces share GLL nodes at face-face edges/corners,
    // so two different quadrature indices may write to the same fluid DOF.
    // The scatter is wrapped in mfem::AtomicAdd, which expands to a hardware
    // atomicAdd on CUDA/HIP and to a plain `+=` on single-threaded host
    // (OpenMP is disabled in SEMSWS), keeping CPU behavior bit-identical.
    // -----------------------------------------------------------------
    if (num_local_quad_ > 0) {
        const int   N     = num_local_quad_;
        constexpr int D   = Dim;
        const auto  u_dev = u_s.Read();
        const auto  sdofs = solid_dofs_.Read();
        const auto  fdofs = fluid_dofs_.Read();
        const auto  nrms  = normals_.Read();
        const auto  jws   = jacobian_w_.Read();
        auto        r_dev = fluid_rhs.ReadWrite();
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
            real_t un = 0.0;
            for (int d = 0; d < D; ++d) {
                un += u_dev[sdofs[i * D + d]] * nrms[i * D + d];
            }
            AtomicAdd(r_dev[fdofs[i]], jws[i] * un);
        });
    }

    // -----------------------------------------------------------------
    // Remote contributions: pack u·n for our solid-owned
    // faces, swap with each peer (fluid-owner on the other side), add
    // received u·n to the local fluid RHS at our fluid-owned face DOFs.
    //
    // Buffers are UseDevice(true); when GPU-aware MPI is active we pass
    // device pointers directly to MPI, otherwise we fall back to host
    // staging via HostRead/HostReadWrite (branch at the MPI call site).
    // -----------------------------------------------------------------
    const bool have_remote =
        !remote_solid_owned_.empty() || !remote_fluid_owned_.empty();
    if (!have_remote) return;

    MPI_Comm comm = parent_ ? parent_->GetComm() : MPI_COMM_WORLD;

    // Pack solid-owned u·n into solid_owned_buf_ on device. One forall
    // per remote face (typical rank has O(1) remote faces so the launch
    // overhead is negligible). byNODES VDOF layout lets us compute
    // vdof = scalar_dof + d*nscalar without a DofToVDof call.
    constexpr int D = Dim;
    const int solid_nscalar = solid_fes_->GetNDofs();
    {
        auto       sbuf_dev = solid_owned_buf_.ReadWrite();
        const auto u_dev    = u_s.Read();
        for (const auto& rec : remote_solid_owned_) {
            const int   N       = rec.my_dofs.Size();
            const int   offset  = rec.buf_offset;
            const auto  sdofs   = rec.my_dofs.Read();
            const auto  nrms    = rec.normals.Read();
            const int   nscalar = solid_nscalar;
            mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
                const int scalar_dof = sdofs[i];
                real_t un = 0.0;
                for (int d = 0; d < D; ++d) {
                    const int vdof = scalar_dof + d * nscalar;
                    un += u_dev[vdof] * nrms[i * D + d];
                }
                sbuf_dev[offset + i] = un;
            });
        }
    }
    // Ensure pack kernels retire before MPI reads the buffer pointer.
    // No-op on CPU builds.
    MFEM_DEVICE_SYNC;

    // Non-blocking send/recv per peer. With GPU-aware MPI we hand MPI
    // device pointers directly (zero-copy); otherwise we stage through
    // host memory. HostRead/HostReadWrite triggers a device→host copy
    // only when the Vector is dirty on device, and the post-Waitall
    // scatter forall's Read() will lazily re-upload received data.
    const bool gpu_aware = mfem::Device::GetGPUAwareMPI();
    const real_t* sbuf = gpu_aware ? solid_owned_buf_.Read()
                                   : solid_owned_buf_.HostRead();
    real_t*       rbuf = gpu_aware ? fluid_owned_buf_.ReadWrite()
                                   : fluid_owned_buf_.HostReadWrite();
    std::vector<MPI_Request> reqs;
    reqs.reserve(peer_ranks_.size() * 2);
    const int npeers = (int)peer_ranks_.size();
    for (int k = 0; k < npeers; ++k) {
        const int p  = peer_ranks_[k];
        const int sc = peer_solid_owned_counts_[k];
        const int rc = peer_fluid_owned_counts_[k];
        if (sc > 0) {
            MPI_Request req;
            MPI_Isend(sbuf + peer_solid_owned_displs_[k], sc, MPIRealT(),
                      p, kTagUn, comm, &req);
            reqs.push_back(req);
        }
        if (rc > 0) {
            MPI_Request req;
            MPI_Irecv(rbuf + peer_fluid_owned_displs_[k], rc, MPIRealT(),
                      p, kTagUn, comm, &req);
            reqs.push_back(req);
        }
    }
    if (!reqs.empty()) {
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }

    // Scatter received u·n into the fluid RHS at our fluid-owned face
    // DOFs on device. Uses AtomicAdd for the same edge/corner-sharing
    // reason as the local scatter above (two remote faces meeting at an
    // element edge can both touch the same fluid DOF).
    {
        const auto rbuf_dev = fluid_owned_buf_.Read();
        auto       r_dev    = fluid_rhs.ReadWrite();
        for (const auto& rec : remote_fluid_owned_) {
            const int   N      = rec.my_dofs.Size();
            const int   offset = rec.buf_offset;
            const auto  fdofs  = rec.my_dofs.Read();
            const auto  jws    = rec.jacobianw.Read();
            mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
                AtomicAdd(r_dev[fdofs[i]],
                                jws[i] * rbuf_dev[offset + i]);
            });
        }
    }
}

template<int Dim>
void FluidSolidInterface<Dim>::ApplyFluidToSolidRHS(
    const ParGridFunction& phi_tt, Vector& solid_rhs)
{
    // Local contributions on device. Uses AtomicAdd for the same
    // edge/corner-sharing reason as ApplySolidToFluidRHS.
    if (num_local_quad_ > 0) {
        const int   N     = num_local_quad_;
        constexpr int D   = Dim;
        const auto  p_dev = phi_tt.Read();
        const auto  sdofs = solid_dofs_.Read();
        const auto  fdofs = fluid_dofs_.Read();
        const auto  nrms  = normals_.Read();
        const auto  jws   = jacobian_w_.Read();
        auto        r_dev = solid_rhs.ReadWrite();
        mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
            const real_t phi_tt_i = p_dev[fdofs[i]];
            for (int d = 0; d < D; ++d) {
                AtomicAdd(r_dev[sdofs[i * D + d]],
                                -jws[i] * phi_tt_i * nrms[i * D + d]);
            }
        });
    }

    // Remote: pack φ_tt on OUR fluid-owned faces, exchange with peers
    // that own the solid on the other side, scatter received φ_tt into
    // the solid RHS on our solid-owned faces.
    const bool have_remote =
        !remote_fluid_owned_.empty() || !remote_solid_owned_.empty();
    if (!have_remote) return;

    MPI_Comm comm = parent_ ? parent_->GetComm() : MPI_COMM_WORLD;
    constexpr int D = Dim;

    // Pack φ_tt from our fluid-owned face DOFs on device.
    {
        auto       sbuf_dev = fluid_owned_buf_.ReadWrite();
        const auto p_dev    = phi_tt.Read();
        for (const auto& rec : remote_fluid_owned_) {
            const int   N      = rec.my_dofs.Size();
            const int   offset = rec.buf_offset;
            const auto  fdofs  = rec.my_dofs.Read();
            mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
                sbuf_dev[offset + i] = p_dev[fdofs[i]];
            });
        }
    }
    // Ensure pack kernels retire before MPI reads the buffer pointer.
    MFEM_DEVICE_SYNC;

    // GPU-aware MPI: zero-copy device pointers. Otherwise stage via host.
    const bool gpu_aware = mfem::Device::GetGPUAwareMPI();
    const real_t* sbuf = gpu_aware ? fluid_owned_buf_.Read()
                                   : fluid_owned_buf_.HostRead();
    real_t*       rbuf = gpu_aware ? solid_owned_buf_.ReadWrite()
                                   : solid_owned_buf_.HostReadWrite();
    std::vector<MPI_Request> reqs;
    reqs.reserve(peer_ranks_.size() * 2);
    const int npeers = (int)peer_ranks_.size();
    for (int k = 0; k < npeers; ++k) {
        const int p  = peer_ranks_[k];
        const int sc = peer_fluid_owned_counts_[k];
        const int rc = peer_solid_owned_counts_[k];
        if (sc > 0) {
            MPI_Request req;
            MPI_Isend(sbuf + peer_fluid_owned_displs_[k], sc, MPIRealT(),
                      p, kTagPhiTt, comm, &req);
            reqs.push_back(req);
        }
        if (rc > 0) {
            MPI_Request req;
            MPI_Irecv(rbuf + peer_solid_owned_displs_[k], rc, MPIRealT(),
                      p, kTagPhiTt, comm, &req);
            reqs.push_back(req);
        }
    }
    if (!reqs.empty()) {
        MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
    }

    // Scatter received φ_tt into the solid RHS at our solid-owned
    // face DOFs (expanded to vector VDOFs per component) on device.
    // byNODES VDOF layout: vdof = scalar_dof + d * nscalar_solid.
    {
        const int  nscalar = solid_fes_->GetNDofs();
        const auto rbuf_dev = solid_owned_buf_.Read();
        auto       r_dev    = solid_rhs.ReadWrite();
        for (const auto& rec : remote_solid_owned_) {
            const int   N      = rec.my_dofs.Size();
            const int   offset = rec.buf_offset;
            const auto  sdofs  = rec.my_dofs.Read();
            const auto  nrms   = rec.normals.Read();
            const auto  jws    = rec.jacobianw.Read();
            mfem::forall(N, [=] MFEM_HOST_DEVICE (int i) {
                const real_t phi_tt_i  = rbuf_dev[offset + i];
                const int    scalar_dof = sdofs[i];
                for (int d = 0; d < D; ++d) {
                    const int vdof = scalar_dof + d * nscalar;
                    AtomicAdd(r_dev[vdof],
                                    -jws[i] * phi_tt_i * nrms[i * D + d]);
                }
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

template class FluidSolidInterface<2>;
template class FluidSolidInterface<3>;

}  // namespace SEM
