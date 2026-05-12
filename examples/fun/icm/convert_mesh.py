"""
Convert HOHQMesh ABAQUS mesh to Gmsh 2.2 format (readable by MFEM).

Uses meshio for both reading and writing.
Parses HOHQMesh's boundary information from ABAQUS comment lines
and creates proper boundary line elements with MFEM-compatible attributes.

SEMSWS 2D boundary attribute convention:
    1 = bottom (y_min)
    2 = right  (x_max)
    3 = top    (y_max)
    4 = left   (x_min)
    5+ = inner boundaries (curve_1, curve_2, ...)

Usage:
    python convert_mesh.py icm.inp icm.msh
"""
import sys
import re
import numpy as np
import meshio


# HOHQMesh boundary name -> SEMSWS attribute mapping
BOUNDARY_ATTR = {
    "Bottom": 1,
    "Right": 2,
    "Top": 3,
    "Left": 4,
}


def parse_hohqmesh_boundaries(inp_file, n_elements):
    """Parse HOHQMesh boundary info from ABAQUS comment lines.

    HOHQMesh writes per-element boundary names as comments:
        ** name_-x name_+x name_-y name_+y
    where '---' means no boundary on that side.

    Returns list of (node1, node2, attribute) for each boundary edge.
    Node indices are 0-based (meshio convention).
    """
    lines = open(inp_file).readlines()

    # Read element connectivity (0-based via meshio later, but here 1-based)
    elements = []
    in_elements = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("*ELEMENT"):
            in_elements = True
            continue
        if stripped.startswith("**"):
            continue
        if stripped.startswith("*"):
            in_elements = False
            continue
        if in_elements:
            parts = [int(p) for p in stripped.split(",")]
            elements.append(parts[1:])  # skip element ID, keep 1-based

    # Parse boundary name lines from comment section
    boundary_name_lines = []
    found_boundary_info = False

    for line in lines:
        stripped = line.strip()
        if "HOHQMesh boundary information" in stripped:
            found_boundary_info = True
            continue
        if not found_boundary_info:
            continue
        if not stripped.startswith("**"):
            continue

        content = stripped[2:].strip()
        # Boundary name lines: 4 words like "Left --- Bottom ---"
        if re.match(r'^[\w-]+(\s+[\w-]+){3}$', content):
            parts = content.split()
            if all(re.match(r'^[A-Za-z_]', p) or p == '---' for p in parts):
                boundary_name_lines.append(parts)

    print(f"  Boundary name lines found: {len(boundary_name_lines)}")

    if len(boundary_name_lines) > n_elements:
        boundary_name_lines = boundary_name_lines[-n_elements:]

    # Inner boundary curve name -> attribute
    inner_curve_attrs = {}
    next_inner_attr = 5

    # HOHQMesh boundary names order: -x +x -y +y
    # Quad nodes (1-based): n1 n2 n3 n4 (CCW from bottom-left)
    #   n4 --- n3
    #   |       |
    #   n1 --- n2
    # Edge mapping:
    #   -x: n4->n1,  +x: n2->n3,  -y: n1->n2,  +y: n3->n4

    boundary_edges = []  # (node1_0based, node2_0based, attribute)

    for eidx, names in enumerate(boundary_name_lines):
        if eidx >= len(elements):
            break
        n1, n2, n3, n4 = elements[eidx]  # 1-based

        side_edges = [
            (names[0], n4, n1),  # -x
            (names[1], n2, n3),  # +x
            (names[2], n1, n2),  # -y
            (names[3], n3, n4),  # +y
        ]

        for name, na, nb in side_edges:
            if name == "---":
                continue
            if name in BOUNDARY_ATTR:
                attr = BOUNDARY_ATTR[name]
            else:
                if name not in inner_curve_attrs:
                    inner_curve_attrs[name] = next_inner_attr
                    next_inner_attr += 1
                attr = inner_curve_attrs[name]
            # Convert to 0-based for meshio
            boundary_edges.append((na - 1, nb - 1, attr))

    print(f"  Boundary edges: {len(boundary_edges)}")
    print(f"  Attribute mapping:")
    all_attrs = {**BOUNDARY_ATTR, **inner_curve_attrs}
    for name, attr in sorted(all_attrs.items(), key=lambda x: x[1]):
        count = sum(1 for _, _, a in boundary_edges if a == attr)
        if count > 0:
            print(f"    {attr}: {name} ({count} edges)")

    return boundary_edges


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.inp> <output.msh>")
        sys.exit(1)

    inp_file = sys.argv[1]
    out_file = sys.argv[2]

    # Read with meshio
    print(f"Reading ABAQUS mesh: {inp_file}")
    mesh = meshio.read(inp_file)
    points = mesh.points
    quads = mesh.cells_dict["quad"]  # 0-based

    print(f"  Points: {len(points)}")
    print(f"  Quads: {len(quads)}")

    # Parse boundary info from HOHQMesh comments
    print(f"\nParsing HOHQMesh boundary information...")
    boundary_edges = parse_hohqmesh_boundaries(inp_file, len(quads))

    # Build meshio Mesh with boundary lines + quads
    bdr_nodes = np.array([[e[0], e[1]] for e in boundary_edges], dtype=np.int64)
    bdr_attrs = np.array([e[2] for e in boundary_edges], dtype=np.int64)
    quad_attrs = np.ones(len(quads), dtype=np.int64)

    cells = [
        meshio.CellBlock("line", bdr_nodes),
        meshio.CellBlock("quad", quads),
    ]
    # Gmsh 2.2 uses cell_data with "gmsh:physical" and "gmsh:geometrical" tags
    cell_data = {
        "gmsh:physical": [bdr_attrs, quad_attrs],
        "gmsh:geometrical": [bdr_attrs, quad_attrs],
    }

    out_mesh = meshio.Mesh(
        points=points,
        cells=cells,
        cell_data=cell_data,
    )

    # Write Gmsh 2.2 via meshio
    print(f"\nWriting Gmsh 2.2: {out_file}")
    meshio.gmsh.write(out_file, out_mesh, fmt_version="2.2", binary=False)

    print(f"  Boundary lines: {len(bdr_nodes)}")
    print(f"  Quad elements: {len(quads)}")

    print("\nDone. Suggested SEMSWS config.yaml boundary section:")
    print("  boundary:")
    print("    absorbing:")
    print("      sides: [bottom, right, top, left]")
    print("    dirichlet:")
    print("      attributes: []")


if __name__ == "__main__":
    main()
