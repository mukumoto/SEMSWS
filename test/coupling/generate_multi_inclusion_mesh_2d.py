"""
Generate a 2D GMSH mesh with TWO disjoint fluid inclusions embedded in a
solid host. Exercises the multi-component topology that MFEM's
`ParSubMesh` handles by assigning a single auto-generated boundary
attribute to all newly-exposed interior faces — regardless of whether
those faces form one connected interface or several.

Layout (y vertical, rectangle [0,Lx] × [0,Lz]):

    ┌──────────────────────────────────────────────┐
    │                                              │
    │        SOLID (attr=2)                        │
    │    ┌──────────┐              ┌──────────┐    │
    │    │  FLUID   │              │  FLUID   │    │
    │    │ (attr=1) │              │ (attr=1) │    │
    │    └──────────┘              └──────────┘    │
    │                                              │
    │                                              │
    └──────────────────────────────────────────────┘

Both fluid inclusions share `attribute=1`, so
`ParSubMesh::CreateFromDomain(parent, {1})` gathers them into ONE
submesh with two disconnected components. The union of their
boundaries with the solid host is the fluid-solid interface Γ_fs,
auto-attributed as `max(parent.bdr_attributes) + 1` by MFEM (same
value for both inclusion boundaries — see `submesh_utils.cpp:318,326`
and `submesh.cpp:182`).

Output: GMSH format 2.2 with one Physical Surface per material plus
a lumped "outer" Physical Line for the four outer rectangle edges.
The interface faces around each inclusion stay intentionally
UNTAGGED so MFEM does the auto-generation (the whole point).
"""

import argparse
import sys

import gmsh


def build_mesh(output: str, lx: float, lz: float,
               mesh_size: float,
               inclusion_w: float, inclusion_h: float,
               inclusion_centers: list[tuple[float, float]]) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("fluid_solid_multi_inclusion_2d")

    # Outer solid host rectangle.
    host = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, lx, lz)

    # Fluid inclusions — built as separate OCC rectangles centered on
    # `inclusion_centers`.
    inclusion_tags = []
    for (cx, cy) in inclusion_centers:
        tag = gmsh.model.occ.addRectangle(
            cx - inclusion_w / 2.0, cy - inclusion_h / 2.0, 0.0,
            inclusion_w, inclusion_h,
        )
        inclusion_tags.append(tag)

    # Fragment: host with every inclusion. After this, the host is split
    # into (solid everywhere-except-inclusions) + (each inclusion
    # interior as its own surface), with conforming shared edges.
    tool_dimtags = [(2, t) for t in inclusion_tags]
    gmsh.model.occ.fragment([(2, host)], tool_dimtags)
    gmsh.model.occ.synchronize()

    # Classify surfaces by centroid: any surface whose centroid lies
    # inside one of the inclusion rectangles is "fluid"; everything
    # else is "solid".
    def centroid(surf_tag: int) -> tuple[float, float]:
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(2, surf_tag)
        return 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)

    def in_any_inclusion(x: float, y: float) -> bool:
        for (cx, cy) in inclusion_centers:
            if (abs(x - cx) <= inclusion_w / 2.0 - 1e-6
                    and abs(y - cy) <= inclusion_h / 2.0 - 1e-6):
                return True
        return False

    fluid_surfs, solid_surfs = [], []
    for _, tag in gmsh.model.getEntities(dim=2):
        cx, cy = centroid(tag)
        if in_any_inclusion(cx, cy):
            fluid_surfs.append(tag)
        else:
            solid_surfs.append(tag)

    assert len(fluid_surfs) == len(inclusion_centers), (
        f"expected {len(inclusion_centers)} fluid surfaces, got {len(fluid_surfs)}"
    )
    assert len(solid_surfs) >= 1, "expected at least one solid surface"

    # Physical groups — same convention as the smoke test:
    #   attribute 1 = fluid (both inclusions, gathered)
    #   attribute 2 = solid (the host)
    gmsh.model.addPhysicalGroup(2, fluid_surfs, tag=1, name="fluid")
    gmsh.model.addPhysicalGroup(2, solid_surfs, tag=2, name="solid")

    # Outer boundary (bottom/right/top/left of the host rectangle).
    # Lump under a single tag — the test doesn't need per-side sides.
    tol = 1e-5 * max(lx, lz)
    outer_edges = []
    for _, tag in gmsh.model.getEntities(1):
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, tag)
        on_outer = (
            (abs(ymin) < tol and abs(ymax) < tol) or
            (abs(ymin - lz) < tol and abs(ymax - lz) < tol) or
            (abs(xmin) < tol and abs(xmax) < tol) or
            (abs(xmin - lx) < tol and abs(xmax - lx) < tol)
        )
        if on_outer:
            outer_edges.append(tag)
    assert outer_edges, "failed to detect outer rectangle edges"
    gmsh.model.addPhysicalGroup(1, outer_edges, tag=11, name="outer")
    # Inclusion boundary edges stay UNTAGGED — MFEM auto-generates the
    # interface attribute for them on the ParSubMesh side.

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    for _, tag in gmsh.model.getEntities(dim=2):
        gmsh.model.mesh.setRecombine(2, tag)
    gmsh.model.mesh.generate(2)

    gmsh.write(output)
    gmsh.finalize()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--lx", type=float, default=1200.0)
    p.add_argument("--lz", type=float, default=800.0)
    p.add_argument("--mesh-size", type=float, default=40.0)
    p.add_argument("--incl-w", type=float, default=200.0)
    p.add_argument("--incl-h", type=float, default=200.0)
    p.add_argument("--centers", type=str,
                   default="300,400;900,400",
                   help="Semicolon-separated x,y centers of fluid "
                        "inclusions (default: two boxes at y=Lz/2).")
    args = p.parse_args()

    centers: list[tuple[float, float]] = []
    for token in args.centers.split(";"):
        x, y = token.split(",")
        centers.append((float(x), float(y)))

    build_mesh(args.output, args.lx, args.lz,
               args.mesh_size, args.incl_w, args.incl_h, centers)

    print(f"NUM_INCLUSIONS={len(centers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
