"""Generate a minimal GMSH mesh with two Physical Volumes for the fluid-solid
coupling probe.

Layout: a box [0, Lx] x [0, Ly] x [0, Lz] split in the middle along z.
  - Lower half (z < Lz/2): Physical Volume "fluid" (tag 1)
  - Upper half (z > Lz/2): Physical Volume "solid" (tag 2)

The interior interface at z = Lz/2 is *not* tagged as a Physical Surface;
the test relies on ParSubMesh auto-generating a new boundary attribute
(max_parent_bdr + 1) for the cut face.

Output: GMSH format 2.2 (matches the SEMSWS .msh pipeline).
"""

import argparse
import sys

import gmsh


def build_mesh(output: str, lx: float, ly: float, lz: float,
               nx: int, ny: int, nz_half: int,
               mfem_bdrs: bool = False) -> None:
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("fluid_solid_probe")

    # Two stacked boxes sharing the mid-plane z = lz/2.
    fluid_tag = gmsh.model.occ.addBox(0, 0, 0, lx, ly, lz / 2)
    solid_tag = gmsh.model.occ.addBox(0, 0, lz / 2, lx, ly, lz / 2)

    # Glue the shared face so the mesh is conforming across z = lz/2.
    gmsh.model.occ.fragment([(3, fluid_tag)], [(3, solid_tag)])
    gmsh.model.occ.synchronize()

    # After fragment, tags may have been renumbered. Re-query the two volumes
    # by their centroids to be safe.
    vols = gmsh.model.getEntities(dim=3)
    assert len(vols) == 2, f"expected 2 volumes after fragment, got {len(vols)}"

    def centroid_z(vol_tag: int) -> float:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(3, vol_tag)
        return 0.5 * (zmin + zmax)

    fluid_vol = min(vols, key=lambda v: centroid_z(v[1]))[1]
    solid_vol = max(vols, key=lambda v: centroid_z(v[1]))[1]

    # Transfinite structured hex mesh so element counts are fully deterministic.
    for dim, tag in gmsh.model.getEntities(1):
        gmsh.model.mesh.setTransfiniteCurve(tag, 2)  # filled in below
    # Set divisions explicitly per curve based on its direction.
    for dim, tag in gmsh.model.getEntities(1):
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, tag)
        dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
        if abs(dx) > max(abs(dy), abs(dz)):
            gmsh.model.mesh.setTransfiniteCurve(tag, nx + 1)
        elif abs(dy) > abs(dz):
            gmsh.model.mesh.setTransfiniteCurve(tag, ny + 1)
        else:
            gmsh.model.mesh.setTransfiniteCurve(tag, nz_half + 1)

    for dim, tag in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(2, tag)

    for dim, tag in gmsh.model.getEntities(3):
        gmsh.model.mesh.setTransfiniteVolume(tag)

    # Physical Volumes — the thing whose inheritance into MFEM attributes we
    # are actually probing.
    gmsh.model.addPhysicalGroup(3, [fluid_vol], tag=1, name="fluid")
    gmsh.model.addPhysicalGroup(3, [solid_vol], tag=2, name="solid")

    # Physical Surfaces — simulate a realistic config that splits the 6
    # outer faces into multiple boundary groups (top / bottom / sides / ABC
    # etc.). This is the common case in SEMSWS configs; Phase 1 auto
    # interface detection must survive non-trivial parent bdr_attrs.
    #
    # Tagging scheme (arbitrary, just exercises multiple attrs):
    #   11 = z=0  (fluid bottom)
    #   12 = z=Lz (solid top)
    #   13 = remaining outer faces (x/y sides of both halves)
    tol = 1e-5 * max(lx, ly, lz)  # gmsh bbox has ~1e-7 absolute slop
    if not mfem_bdrs:
        # Default: lump sides together into one tag to stress auto-
        # detection over non-trivial parent bdrs (used by most tests).
        bottom_faces, top_faces, side_faces = [], [], []
        for dim, tag in gmsh.model.getEntities(2):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
            if abs(zmin) < tol and abs(zmax) < tol:
                bottom_faces.append(tag)
            elif abs(zmin - lz) < tol and abs(zmax - lz) < tol:
                top_faces.append(tag)
            elif abs(zmin - lz / 2) < tol and abs(zmax - lz / 2) < tol:
                pass  # interior interface — intentionally NOT tagged
            else:
                side_faces.append(tag)
        if bottom_faces:
            gmsh.model.addPhysicalGroup(2, bottom_faces, tag=11, name="bottom")
        if top_faces:
            gmsh.model.addPhysicalGroup(2, top_faces, tag=12, name="top")
        if side_faces:
            gmsh.model.addPhysicalGroup(2, side_faces, tag=13, name="sides")
    else:
        # MFEM MakeCartesian3D convention for use with Cerjan side names
        # (ParseBoundarySide maps "bottom"→1, "front"→2, "right"→3,
        # "back"→4, "left"→5, "top"→6).
        faces_by_tag = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
        for dim, tag in gmsh.model.getEntities(2):
            xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(2, tag)
            if abs(zmin) < tol and abs(zmax) < tol:
                faces_by_tag[1].append(tag)                    # bottom
            elif abs(ymin) < tol and abs(ymax) < tol:
                faces_by_tag[2].append(tag)                    # front (y=0)
            elif abs(xmin - lx) < tol and abs(xmax - lx) < tol:
                faces_by_tag[3].append(tag)                    # right (x=Lx)
            elif abs(ymin - ly) < tol and abs(ymax - ly) < tol:
                faces_by_tag[4].append(tag)                    # back (y=Ly)
            elif abs(xmin) < tol and abs(xmax) < tol:
                faces_by_tag[5].append(tag)                    # left (x=0)
            elif abs(zmin - lz) < tol and abs(zmax - lz) < tol:
                faces_by_tag[6].append(tag)                    # top
            elif abs(zmin - lz / 2) < tol and abs(zmax - lz / 2) < tol:
                pass  # interior interface
        names = {1: "bottom", 2: "front", 3: "right",
                 4: "back", 5: "left", 6: "top"}
        for ttag, fs in faces_by_tag.items():
            if fs:
                gmsh.model.addPhysicalGroup(2, fs, tag=ttag, name=names[ttag])

    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(3)

    gmsh.write(output)
    gmsh.finalize()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--lx", type=float, default=100.0)
    p.add_argument("--ly", type=float, default=100.0)
    p.add_argument("--lz", type=float, default=200.0)
    p.add_argument("--nx", type=int, default=2)
    p.add_argument("--ny", type=int, default=2)
    p.add_argument("--nz-half", type=int, default=2,
                   help="Elements per half in z (fluid gets nz_half, solid gets nz_half)")
    p.add_argument("--mfem-bdrs", action="store_true",
                   help="Tag each outer face separately with MFEM MakeCartesian3D "
                        "numbering (bottom=1, front=2, right=3, back=4, left=5, top=6). "
                        "Required for configs that drive Cerjan through named "
                        "`boundary.absorbing.sides`.")
    args = p.parse_args()

    build_mesh(args.output, args.lx, args.ly, args.lz,
               args.nx, args.ny, args.nz_half, args.mfem_bdrs)

    # Expected element counts: each box has nx * ny * nz_half hexes.
    per_box = args.nx * args.ny * args.nz_half
    print(f"EXPECTED_FLUID_NE={per_box}")
    print(f"EXPECTED_SOLID_NE={per_box}")
    print(f"EXPECTED_TOTAL_NE={2 * per_box}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
