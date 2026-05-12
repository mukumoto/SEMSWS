"""Visualize the generated HOHQMesh ABAQUS mesh."""
import numpy as np
import matplotlib.pyplot as plt
import sys


def parse_abaqus(filename):
    """Parse ABAQUS .inp file for nodes and elements."""
    nodes = {}
    elements = []
    section = None

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("*NODE"):
                section = "node"
                continue
            elif line.startswith("*ELEMENT"):
                section = "element"
                continue
            elif line.startswith("*"):
                section = None
                continue

            if section == "node":
                parts = line.split(",")
                nid = int(parts[0])
                x, y = float(parts[1]), float(parts[2])
                nodes[nid] = (x, y)
            elif section == "element":
                parts = [int(p) for p in line.split(",")]
                eid = parts[0]
                conn = parts[1:]
                elements.append(conn)

    return nodes, elements


def main():
    fname = sys.argv[1] if len(sys.argv) > 1 else "icm.inp"
    nodes, elements = parse_abaqus(fname)

    print(f"Nodes: {len(nodes)}, Elements: {len(elements)}")

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    # Plot elements as quad outlines
    for conn in elements:
        pts = [nodes[n] for n in conn if n in nodes]
        if len(pts) >= 4:
            # Take corners (first 4 for linear quad)
            corners = pts[:4] + [pts[0]]
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            ax.plot(xs, ys, "b-", linewidth=0.3, alpha=0.6)

    # Plot source position if available
    try:
        src = np.loadtxt("source_position.txt")
        ax.plot(src[0], src[1], "r*", markersize=15, label="Source (i-dot)")
        ax.legend()
    except Exception:
        pass

    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("ICM mesh (HOHQMesh ABAQUS output)")
    plt.tight_layout()
    plt.savefig("icm_mesh.png", dpi=150)
    print("Saved: icm_mesh.png")
    plt.close()


if __name__ == "__main__":
    main()
