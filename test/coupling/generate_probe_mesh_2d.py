"""
Generate a minimal GMSH 2D mesh with two Physical Surfaces for the
fluid-solid coupling probe. 2D analogue of generate_probe_mesh.py
(which produces 3D hex meshes).

Layout: a rectangle [0, Lx] x [0, Lz] split in the middle along z.
  - Lower half (z < Lz/2): Physical Surface "fluid" (tag 1)
  - Upper half (z > Lz/2): Physical Surface "solid" (tag 2)

The interior interface at z = Lz/2 is *not* tagged as a Physical Line;
the coupled facade relies on ParSubMesh auto-generating a new boundary
attribute (max_parent_bdr + 1) for the cut edge. Parent boundary tags
on the four outer edges are emitted according to MFEM's MakeCartesian2D
ordering (bottom=1, right=2, top=3, left=4) when --mfem-bdrs is set;
otherwise the four edges are lumped together.

Output: GMSH format 2.2 (matches SEMSWS's existing .msh pipeline).
"""

import argparse
import sys

import gmsh


def build_mesh(output: str, lx: float, lz: float,
               nx: int, nz_half: int,
               mfem_bdrs: bool = False) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("fluid_solid_probe_2d")

    # Two stacked rectangles sharing the mid-line z = lz/2.
    fluid_tag = gmsh.model.occ.addRectangle(0.0, 0.0,       0.0, lx, lz / 2)
    solid_tag = gmsh.model.occ.addRectangle(0.0, lz / 2.0,  0.0, lx, lz / 2)

    # Fragment so the shared edge at z=lz/2 is conforming (same line tag
    # on both sides, making the mesh genuinely 2-manifold).
    gmsh.model.occ.fragment([(2, fluid_tag)], [(2, solid_tag)])
    gmsh.model.occ.synchronize()

    # After fragment, surface tags may have been renumbered — re-find
    # fluid vs solid by centroid z. (Same idempotent pattern as the 3D
    # probe generator.)
    surfs = gmsh.model.getEntities(dim=2)
    assert len(surfs) == 2, (
        f"expected 2 surfaces after fragment, got {len(surfs)}")

    def centroid_z(surf_tag: int) -> float:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, surf_tag)
        # 2D OCC lives in the z=0 plane, so "z" here is actually the
        # second Cartesian coord returned by getBoundingBox, which OCC
        # puts in `ymax/ymin` for 2D rectangles. Use y.
        return 0.5 * (ymin + ymax)

    fluid_surf = min(surfs, key=lambda s: centroid_z(s[1]))[1]
    solid_surf = max(surfs, key=lambda s: centroid_z(s[1]))[1]

    # Transfinite structured quad mesh — element counts fully
    # deterministic, same idea as the 3D generator.
    for dim, tag in gmsh.model.getEntities(1):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, tag)
        dx, dy = xmax - xmin, ymax - ymin
        # x-edges get nx+1 nodes; z-edges get nz_half+1 nodes.
        if abs(dx) > abs(dy):
            gmsh.model.mesh.setTransfiniteCurve(tag, nx + 1)
        else:
            gmsh.model.mesh.setTransfiniteCurve(tag, nz_half + 1)

    for dim, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(2, tag)

    # Physical Surfaces — the two attributes the coupled facade will
    # consume as `material.fluid.attribute=1` / `material.solid.attribute=2`.
    gmsh.model.addPhysicalGroup(2, [fluid_surf], tag=1, name="fluid")
    gmsh.model.addPhysicalGroup(2, [solid_surf], tag=2, name="solid")

    # Physical Lines (boundary edges). Two modes, same pattern as the 3D
    # generator:
    #   mfem_bdrs=False → lump outer edges under a single tag 13 "sides".
    #   mfem_bdrs=True  → emit per-side tags in MFEM's Cartesian2D order
    #                     so YAML configs can refer to named sides
    #                     (bottom/right/top/left) and ParseBoundarySide
    #                     resolves them to attrs 1/2/3/4.
    tol = 1e-5 * max(lx, lz)
    if not mfem_bdrs:
        bottom_edges, top_edges, side_edges = [], [], []
        for dim, tag in gmsh.model.getEntities(1):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, tag)
            if abs(ymin) < tol and abs(ymax) < tol:
                bottom_edges.append(tag)
            elif abs(ymin - lz) < tol and abs(ymax - lz) < tol:
                top_edges.append(tag)
            elif abs(ymin - lz / 2) < tol and abs(ymax - lz / 2) < tol:
                pass  # interior fluid-solid interface, intentionally untagged
            else:
                side_edges.append(tag)
        if bottom_edges:
            gmsh.model.addPhysicalGroup(1, bottom_edges, tag=11, name="bottom")
        if top_edges:
            gmsh.model.addPhysicalGroup(1, top_edges, tag=12, name="top")
        if side_edges:
            gmsh.model.addPhysicalGroup(1, side_edges, tag=13, name="sides")
    else:
        # MFEM MakeCartesian2D convention: bottom=1, right=2, top=3, left=4.
        # (See MFEM source + the "mfem-bdrs" path in the 3D generator.)
        edges_by_tag: dict[int, list[int]] = {1: [], 2: [], 3: [], 4: []}
        for dim, tag in gmsh.model.getEntities(1):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, tag)
            if abs(ymin) < tol and abs(ymax) < tol:
                edges_by_tag[1].append(tag)                    # bottom
            elif abs(xmin - lx) < tol and abs(xmax - lx) < tol:
                edges_by_tag[2].append(tag)                    # right
            elif abs(ymin - lz) < tol and abs(ymax - lz) < tol:
                edges_by_tag[3].append(tag)                    # top
            elif abs(xmin) < tol and abs(xmax) < tol:
                edges_by_tag[4].append(tag)                    # left
            elif abs(ymin - lz / 2) < tol and abs(ymax - lz / 2) < tol:
                pass  # interior interface
        names = {1: "bottom", 2: "right", 3: "top", 4: "left"}
        for ttag, es in edges_by_tag.items():
            if es:
                gmsh.model.addPhysicalGroup(1, es, tag=ttag, name=names[ttag])

    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(2)

    gmsh.write(output)
    gmsh.finalize()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--lx", type=float, default=100.0)
    p.add_argument("--lz", type=float, default=200.0)
    p.add_argument("--nx", type=int, default=4)
    p.add_argument("--nz-half", type=int, default=4,
                   help="Elements per half in z (fluid gets nz_half, "
                        "solid gets nz_half)")
    p.add_argument("--mfem-bdrs", action="store_true",
                   help="Tag each outer edge separately with MFEM "
                        "MakeCartesian2D numbering (bottom=1, right=2, "
                        "top=3, left=4). Required for configs that drive "
                        "Cerjan / Dirichlet through named "
                        "`boundary.*.sides`.")
    args = p.parse_args()

    build_mesh(args.output, args.lx, args.lz,
               args.nx, args.nz_half, args.mfem_bdrs)

    per_surf = args.nx * args.nz_half
    print(f"EXPECTED_FLUID_NE={per_surf}")
    print(f"EXPECTED_SOLID_NE={per_surf}")
    print(f"EXPECTED_TOTAL_NE={2 * per_surf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
