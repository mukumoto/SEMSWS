#!/usr/bin/env python3
"""
Plot GMT xyz wavefield and material files using matplotlib.

Reads grid_info.txt for exact grid parameters.
Handles constant fields (e.g., homogeneous Vp) with appropriate colorbar precision.

Supports two layouts:

* **Pure single-domain** (default). Reads `<results>/material/gmt/` and
  `<results>/wavefield/gmt/`.
* **Coupled fluid-solid** (auto-detected or `--coupled`). Reads
  `<results>/fluid/{material,wavefield}/gmt/` AND
  `<results>/solid/{material,wavefield}/gmt/` and overlays both domains
  on the same axes. Each domain is independently normalised to its own
  ±max(|Z|) so both arrivals are visible despite the Pa-vs-um amplitude
  gap; two inset colorbars report the raw units per domain.

Usage:
    python plot_wavefield_2d.py [--results-dir ./results] [--coupled]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np


# --------------------------------------------------------------------------
# Common loaders
# --------------------------------------------------------------------------


def read_grid_info(info_path):
    """Read grid_info.txt and return dict of parameters."""
    info = {}
    with open(info_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            info[key] = val
    return info


def load_xyz(filepath, grid_info=None):
    """Load GMT xyz file using grid_info for exact reshape.

    If `grid_info`'s (nx, ny) don't match the file's row count (e.g.
    grid_info.txt is from a different run or a different bounding box),
    fall back to detecting the grid dimensions from the coordinate stream."""
    data = np.loadtxt(filepath)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    n = len(x)

    nx = ny = None
    if grid_info:
        try:
            nx = int(grid_info["nx"])
            ny = int(grid_info["ny"])
            if nx * ny != n:
                # grid_info stale → fall through to auto-detection
                nx = ny = None
        except (KeyError, ValueError):
            nx = ny = None

    if nx is None:
        # Auto-detect: find the first x-index where x wraps around
        # (column-major xyz: x varies fastest inside each row).
        detected = 1
        for i in range(1, n):
            if x[i] <= x[i - 1]:
                detected = i
                break
        nx = detected
        ny = n // nx

    X = x.reshape(ny, nx)
    Y = y.reshape(ny, nx)
    Z = z.reshape(ny, nx)

    return X, Y, Z


def is_constant_field(Z, rtol=1e-4):
    """Check if field is approximately constant (e.g., homogeneous material)."""
    valid = Z[~np.isnan(Z)]
    if len(valid) == 0:
        return True, 0.0
    mean_val = np.mean(valid)
    if mean_val == 0:
        return np.all(valid == 0), 0.0
    rel_range = (np.max(valid) - np.min(valid)) / abs(mean_val)
    return rel_range < rtol, mean_val


# --------------------------------------------------------------------------
# GMSH .msh (version 2.2) parser — used to paint fluid/solid backgrounds
# in the coupled plotting path, same helper as make_animation.py.
# --------------------------------------------------------------------------


def parse_gmsh22(msh_file: str):
    """Parse a GMSH 2.2 ASCII mesh and return
    `(nodes: dict[int, (x, y)], quads: list[(attr, [n1, n2, n3, n4])])`.

    Each quad's `attr` is its Physical Group tag (1 = fluid, 2 = solid
    in generate_mesh.py's convention)."""
    nodes: dict[int, tuple[float, float]] = {}
    quads: list[tuple[int, list[int]]] = []
    section = None
    with open(msh_file) as f:
        for line in f:
            s = line.strip()
            if s.startswith("$"):
                if s == "$Nodes":    section = "nodes"
                elif s == "$Elements": section = "elements"
                elif s.startswith("$End"): section = None
                else: section = None
                continue
            if section == "nodes":
                parts = s.split()
                if len(parts) == 4:
                    nodes[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif section == "elements":
                parts = s.split()
                if len(parts) < 5:
                    continue
                try:
                    etype = int(parts[1])
                except ValueError:
                    continue
                if etype != 3:
                    continue
                ntags = int(parts[2])
                phys = int(parts[3])
                node_start = 3 + ntags
                conn = [int(p) for p in parts[node_start:node_start + 4]]
                quads.append((phys, conn))
    return nodes, quads


def quad_polys_and_edges(nodes: dict, quads):
    """Separate quads by physical attribute; return
    `(fluid_polys, solid_polys, edges)`."""
    fluid_polys: list[list[tuple[float, float]]] = []
    solid_polys: list[list[tuple[float, float]]] = []
    edges: list[list[tuple[float, float]]] = []
    for attr, conn in quads:
        pts = [nodes[n] for n in conn]
        if attr == 1:
            fluid_polys.append(pts)
        else:
            solid_polys.append(pts)
        for i in range(4):
            edges.append([pts[i], pts[(i + 1) % 4]])
    return fluid_polys, solid_polys, edges


def _alpha_seismic_cmap(alpha_peak: float = 0.9):
    """Return a seismic colormap whose alpha ramps symmetrically from
    0 at the midpoint (value=0 in [-1, 1] normalisation) to `alpha_peak`
    at the extremes. Only used on the per-frame path; the global path
    uses `_wave_rgba` for decoupled colour/alpha scales. NaN → transparent."""
    base = plt.get_cmap("seismic")
    colors = base(np.linspace(0, 1, 256))
    v = np.linspace(0, 1, 256)
    colors[:, 3] = np.abs(2 * (v - 0.5)) * alpha_peak
    cm = mcolors.ListedColormap(colors)
    cm.set_bad(color=(0, 0, 0, 0))
    return cm


def _wave_rgba(Z: np.ndarray, vmax_color: float, vmax_alpha: float,
               alpha_peak: float = 0.9,
               cmap_name: str = "seismic",
               mode: str = "signed",
               gamma: float = 1.0) -> np.ndarray:
    """Build an RGBA image with colour + alpha on decoupled scales.

    Colour normalisation depends on `mode`:
      * `"signed"`     — Z normalised into [-1, 1], cmap sampled at
                         (nz+1)/2. Use with diverging cmaps like seismic.
      * `"magnitude"`  — |Z| normalised into [0, 1], cmap sampled at
                         t = |Z|/vmax_color. Use with sequential cmaps
                         like turbo/inferno/magma/viridis.

    Alpha is always tied to `|Z| / vmax_alpha` so low-amplitude frames
    under global colour normalisation still show their peak as
    opaque-ish. NaN → fully transparent."""
    if vmax_color <= 0: vmax_color = 1.0
    if vmax_alpha <= 0: vmax_alpha = 1.0
    cmap = plt.get_cmap(cmap_name)
    if mode == "magnitude":
        t = np.clip(np.abs(Z) / vmax_color, 0.0, 1.0)
        rgba = cmap(t)
    else:
        nz_color = np.clip(Z / vmax_color, -1.0, 1.0)
        rgba = cmap((nz_color + 1.0) * 0.5)
    # Alpha: magnitude-normalised per-frame, capped at alpha_peak.
    a = np.clip(np.abs(Z) / vmax_alpha, 0.0, 1.0) * alpha_peak
    rgba[..., 3] = a
    # NaN → fully transparent (matplotlib would propagate NaN through
    # the colormap call above and the alpha clip, so zero out explicitly).
    mask = ~np.isfinite(Z)
    rgba[mask] = (0.0, 0.0, 0.0, 0.0)
    return rgba


def _add_mesh_background(ax, mesh_info, fluid_bg: str, solid_bg: str,
                        show_edges: bool):
    """Paint fluid & solid polygon backgrounds + optional mesh edges on
    `ax`. `mesh_info` is the tuple returned by `quad_polys_and_edges`,
    or None to skip."""
    if mesh_info is None:
        return
    fluid_polys, solid_polys, edges = mesh_info
    if solid_polys:
        ax.add_collection(PolyCollection(
            solid_polys, facecolors=solid_bg, edgecolors="none",
            zorder=0))
    if fluid_polys:
        ax.add_collection(PolyCollection(
            fluid_polys, facecolors=fluid_bg, edgecolors="none",
            zorder=1))
    if show_edges:
        ax.add_collection(LineCollection(
            edges, colors="black", linewidths=0.15, alpha=0.3, zorder=2))


def parse_exodus_quads(exo_file: str):
    """Read an Exodus (.exo/.e/.gen) mesh via meshio and return
    `(nodes: dict[int, (x, y)], quads: list[(attr, [n1, n2, n3, n4])])`
    in the same shape as `parse_gmsh22`. Exodus blocks map to `attr` —
    block 1 → fluid (attr=1), block 2 → solid (attr=2), matching the
    SEMSWS coupled convention produced by generate_jou.py."""
    import meshio  # only needed when the user points at an Exodus file
    m = meshio.read(exo_file)
    pts = m.points  # (N, 2) or (N, 3) — we keep x,y only
    nodes: dict[int, tuple[float, float]] = {
        i: (float(pts[i, 0]), float(pts[i, 1])) for i in range(len(pts))
    }
    quads: list[tuple[int, list[int]]] = []
    # Exodus blocks are typically preserved as separate cell_sets in
    # meshio. We read them by block name if present, falling back to
    # cell-block order (first quad block → fluid, second → solid).
    fluid_keys = {"fluid", "Block 1", "block_1", "block-1", "BLOCK_1", "1"}
    solid_keys = {"solid", "Block 2", "block_2", "block-2", "BLOCK_2", "2"}
    for block_idx, cb in enumerate(m.cells):
        if cb.type not in ("quad", "quad4"):
            continue
        # Find this cellblock's Exodus block id from cell_sets.
        attr: int | None = None
        for name, mask_lists in (m.cell_sets or {}).items():
            if block_idx >= len(mask_lists) or mask_lists[block_idx] is None:
                continue
            if len(mask_lists[block_idx]) == 0:
                continue
            nlow = name.lower()
            if name in fluid_keys or nlow == "fluid":
                attr = 1; break
            if name in solid_keys or nlow == "solid":
                attr = 2; break
        if attr is None:
            # Fallback: preserve block order. First quad cellblock = fluid.
            attr = 1 if block_idx == 0 else 2
        for conn in cb.data:
            quads.append((attr, [int(n) for n in conn]))
    return nodes, quads


def _load_mesh_info(mesh_path: Path | None):
    """Parse mesh (GMSH 2.2 or Exodus) if given, return (fluid_polys,
    solid_polys, edges) or None."""
    if mesh_path is None or not mesh_path.is_file():
        return None
    ext = mesh_path.suffix.lower()
    if ext in (".exo", ".e", ".gen"):
        nodes, quads = parse_exodus_quads(str(mesh_path))
    else:
        nodes, quads = parse_gmsh22(str(mesh_path))
    return quad_polys_and_edges(nodes, quads)


def _autodetect_mesh(results_dir: Path) -> Path | None:
    """Look for the mesh file referenced by `coupled.yaml`, else the
    first `*.msh` sitting next to `results/` (generate_mesh.py's
    default layout)."""
    import yaml
    parent = results_dir.parent
    # Try coupled.yaml then config.yaml.
    for yaml_name in ("coupled.yaml", "config.yaml"):
        y = parent / yaml_name
        if y.is_file():
            try:
                cfg = yaml.safe_load(y.read_text())
                mf = cfg.get("mesh", {}).get("file")
                if mf:
                    p = (parent / mf).resolve()
                    if p.is_file():
                        return p
            except Exception:
                pass
    # Fallback: first *.msh sibling of results_dir.
    msh = sorted(parent.glob("*.msh"))
    return msh[0] if msh else None


# --------------------------------------------------------------------------
# Pure single-domain plotting (unchanged behavior)
# --------------------------------------------------------------------------


def plot_material(results_dir, outdir):
    """Plot material fields (vp, rho)."""
    gmt_dir = results_dir / "material" / "gmt"
    info_path = gmt_dir / "grid_info.txt"

    if not gmt_dir.exists():
        print("  No material GMT directory found")
        return

    grid_info = read_grid_info(info_path) if info_path.exists() else None

    fields = {}
    for f in sorted(gmt_dir.glob("*.xyz")):
        name = f.stem
        X, Y, Z = load_xyz(f, grid_info)
        fields[name] = (X, Y, Z)

    if not fields:
        return

    nfields = len(fields)
    fig, axes = plt.subplots(1, nfields, figsize=(6 * nfields, 5))
    if nfields == 1:
        axes = [axes]

    for ax, (name, (X, Y, Z)) in zip(axes, fields.items()):
        constant, mean_val = is_constant_field(Z)

        if constant:
            # Constant field: fix colorbar to show the single value cleanly
            margin = abs(mean_val) * 0.01 if mean_val != 0 else 1.0
            im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis",
                               vmin=mean_val - margin, vmax=mean_val + margin)
            cb = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[mean_val])
            cb.ax.set_yticklabels([f"{mean_val:g}"])
        else:
            im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
            cb = plt.colorbar(im, ax=ax, shrink=0.8)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(name)
        ax.set_aspect("equal")

    plt.tight_layout()
    outfile = outdir / f"material.{_OUT_EXT}"
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Material -> {outfile}")


MAX_PANELS_PER_PAGE = 12  # max panels per PNG

# Output image extension ("png" or "pdf"). Set from the CLI `--format`
# option in main() before any plotting runs. matplotlib's `savefig`
# picks the rasteriser vs. vector backend from the extension alone,
# so a single module-level string is enough to flip every output
# between bitmap (PNG) and vector (PDF) without plumbing an `ext`
# argument through every function signature.
_OUT_EXT = "png"


def split_evenly(n, max_per_page):
    """Split n items into pages of roughly equal size, each <= max_per_page."""
    if n <= max_per_page:
        return [n]
    n_pages = -(-n // max_per_page)  # ceil division
    per_page = -(-n // n_pages)      # ceil division for even split
    pages = []
    remaining = n
    for _ in range(n_pages):
        count = min(per_page, remaining)
        pages.append(count)
        remaining -= count
    return pages


def plot_snapshot_page(snapshots, suptitle, outfile, vmax_fixed=None,
                       cmap_name="seismic", mode="signed",
                       percentile: float = 100.0):
    """Plot a single page of wavefield snapshots.

    `vmax_fixed` (optional): symmetric ±vmax for every panel. If None,
    each panel is normalised by its own |Z|.max (per-frame).
    `cmap_name` + `mode`: palette + colour-scale semantics. In
    `"magnitude"` mode Z is replaced by |Z| and the colorbar spans
    [0, vmax]; pair with sequential cmaps (turbo / inferno / ...)."""
    ncols = min(4, len(snapshots))
    nrows = -(-len(snapshots) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, (step, X, Y, Z) in enumerate(snapshots):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        if vmax_fixed is not None:
            vmax = vmax_fixed
        else:
            valid = Z[~np.isnan(Z)]
            if len(valid) == 0:
                vmax = 1.0
            elif percentile >= 100.0 or percentile <= 0.0:
                vmax = float(np.max(np.abs(valid)))
            else:
                vmax = float(np.percentile(np.abs(valid), percentile))
        if vmax <= 0:
            vmax = 1.0

        if mode == "magnitude":
            im = ax.pcolormesh(X, Y, np.abs(Z), shading="auto",
                               cmap=cmap_name, vmin=0, vmax=vmax)
        else:
            im = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap_name,
                               vmin=-vmax, vmax=vmax)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Step {step}")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.7)

    for idx in range(len(snapshots), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.suptitle(suptitle, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()


def plot_wavefield(results_dir, outdir, normalize: str = "per-frame",
                   cmap_name: str = "seismic", mode: str = "signed",
                   skip: int = 1, percentile: float = 100.0):
    """Plot wavefield snapshots. `normalize` is one of {per-frame, global}:
    per-frame uses |Z|.max per panel; global uses the max across ALL
    snapshots (constant scale, amplitude decay visible). `cmap_name`
    and `mode` (signed / magnitude) are passed through to
    `plot_snapshot_page`. `skip`: stride applied after the global
    vmax pass so constant-scale plots still use the full-run peak.
    `percentile < 100`: clip vmax against that percentile of |Z|
    instead of the true max (seismic-style gain boost)."""
    gmt_dir = results_dir / "wavefield" / "gmt"
    info_path = gmt_dir / "grid_info.txt"

    if not gmt_dir.exists():
        print("  No wavefield GMT directory found")
        return

    grid_info = read_grid_info(info_path) if info_path.exists() else None

    xyz_files = sorted(gmt_dir.glob("PS_*.xyz"))
    if not xyz_files:
        print("  No wavefield GMT files found")
        return

    snapshots = []
    for f in xyz_files:
        step = f.stem.split("_")[1]
        X, Y, Z = load_xyz(f, grid_info)
        snapshots.append((step, X, Y, Z))

    # Pre-compute global vmax if requested (over FULL snapshot set,
    # before `skip` strides it for plotting). Percentile clipping
    # aggregates every frame's valid |Z| and takes the percentile
    # across the concatenation, matching the coupled path.
    vmax_fixed = None
    if normalize == "global":
        if percentile >= 100.0 or percentile <= 0.0:
            v = 0.0
            for _, _, _, Z in snapshots:
                valid = Z[~np.isnan(Z)]
                if len(valid):
                    v = max(v, float(np.max(np.abs(valid))))
        else:
            chunks = []
            for _, _, _, Z in snapshots:
                valid = Z[~np.isnan(Z)]
                if len(valid):
                    chunks.append(np.abs(valid))
            v = (float(np.percentile(np.concatenate(chunks), percentile))
                 if chunks else 0.0)
        vmax_fixed = v if v > 0 else 1.0

    if skip > 1:
        snapshots = snapshots[::skip]

    pages = split_evenly(len(snapshots), MAX_PANELS_PER_PAGE)
    offset = 0
    for page_idx, page_count in enumerate(pages):
        page_snapshots = snapshots[offset:offset + page_count]
        offset += page_count

        if len(pages) == 1:
            suptitle = "Pressure Field (PS)"
            outfile = outdir / f"wavefield_snapshots.{_OUT_EXT}"
        else:
            suptitle = f"Pressure Field (PS) ({page_idx + 1}/{len(pages)})"
            outfile = outdir / f"wavefield_snapshots_{page_idx + 1}.{_OUT_EXT}"

        plot_snapshot_page(page_snapshots, suptitle, outfile,
                           vmax_fixed=vmax_fixed,
                           cmap_name=cmap_name, mode=mode,
                           percentile=percentile)
        print(f"  Wavefield -> {outfile}")


# --------------------------------------------------------------------------
# Coupled fluid-solid plotting
# --------------------------------------------------------------------------


def is_coupled_layout(results_dir: Path) -> bool:
    """A coupled run puts its output under <results>/fluid/ and
    <results>/solid/. We consider the layout 'coupled' iff either
    subdirectory contains a gmt/ folder."""
    for side in ("fluid", "solid"):
        sd = results_dir / side
        if (sd / "wavefield" / "gmt").is_dir() or (sd / "material" / "gmt").is_dir():
            return True
    return False


def _list_field_prefixes(gmt_dir: Path) -> list[str]:
    """Enumerate all wavefield prefixes present in a GMT directory.

    Filenames are `<prefix>_NNNNNN.xyz` where NNNNNN is a zero-padded
    6-digit step. The prefix is whatever comes before the trailing
    `_NNNNNN`. Examples: `PS`, `DISP`, `DISP_mag`, `DISP_x`, `VEL_mag`.
    Returns them sorted for deterministic plot ordering."""
    prefixes: set[str] = set()
    for f in gmt_dir.glob("*_[0-9][0-9][0-9][0-9][0-9][0-9].xyz"):
        stem = f.stem
        # Trim the 6-digit step and the underscore that precedes it.
        if len(stem) > 7 and stem[-7] == "_" and stem[-6:].isdigit():
            prefixes.add(stem[:-7])
    return sorted(prefixes)


def _load_snapshot_dict(gmt_dir: Path, prefix: str) -> dict[str, tuple]:
    """Load all snapshots under gmt_dir matching `<prefix>_NNNNNN.xyz`
    keyed by the step string."""
    info_path = gmt_dir / "grid_info.txt"
    grid_info = read_grid_info(info_path) if info_path.exists() else None
    out: dict[str, tuple] = {}
    for f in sorted(gmt_dir.glob(f"{prefix}_[0-9]*.xyz")):
        step = f.stem[len(prefix) + 1:]
        X, Y, Z = load_xyz(f, grid_info)
        out[step] = (X, Y, Z)
    return out


def _symmetric_vmax(Z: np.ndarray, percentile: float = 100.0) -> float:
    """Symmetric |Z| upper bound. `percentile=100` → true max (default,
    backwards-compatible); values < 100 clip against an intensity
    percentile so that a single source spike can't flatten every other
    frame into near-transparency. Typical 'percentile clipping' values
    in seismic imaging: 99.5 (gentle), 95 (aggressive)."""
    valid = Z[~np.isnan(Z)]
    if len(valid) == 0:
        return 1.0
    if percentile >= 100.0 or percentile <= 0.0:
        v = float(np.max(np.abs(valid)))
    else:
        v = float(np.percentile(np.abs(valid), percentile))
    return v if v > 0 else 1.0


def _overlay_two_domains(ax, fluid, solid,
                         fluid_label: str, solid_label: str,
                         vmax_f_fixed: float | None = None,
                         vmax_s_fixed: float | None = None,
                         mesh_info=None,
                         fluid_bg: str = "#b8dff5",
                         solid_bg: str = "#d0d0d0",
                         show_edges: bool = False,
                         cmap=None,
                         cmap_name: str = "seismic",
                         mode: str = "signed",
                         percentile: float = 100.0):
    """Draw fluid + solid on the same axes as a SINGLE connected
    wavefield: one colormap, one colorbar, per-domain independent
    normalisation so both arrivals are visible despite the Pa-vs-µm
    scale gap.

    The two submesh grids do NOT have to overlap or share resolution:
    each domain is NaN outside its own submesh, so two pcolormesh calls
    using the same colormap and the same vmin/vmax = ±1 visually stitch
    into one continuous ±1 field.

    Normalisation:
      * If `vmax_{f,s}_fixed` is None → per-frame per-domain (Z/|Z|.max)
      * If given → use that value as the domain's symmetric vmax,
        e.g. a cross-timestep global-per-domain maximum computed by
        the caller. Values exceeding vmax saturate to ±1 in the
        colorbar."""
    # Background (solid gray, fluid light blue) + optional mesh edges,
    # painted first so the wavefield overlays them with the background
    # showing through quiet regions.
    _add_mesh_background(ax, mesh_info, fluid_bg, solid_bg, show_edges)

    Xf, Yf, Zf = fluid
    Xs, Ys, Zs = solid
    # Colour normalisation (per-frame by default, per-domain-global
    # when caller supplies vmax_{f,s}_fixed). Per-frame applies the
    # same percentile clipping the caller requested — keeps a single
    # pixel spike from compressing the dynamic range of the frame.
    vmax_f = vmax_f_fixed if vmax_f_fixed is not None else _symmetric_vmax(Zf, percentile)
    vmax_s = vmax_s_fixed if vmax_s_fixed is not None else _symmetric_vmax(Zs, percentile)
    if vmax_f <= 0: vmax_f = 1.0
    if vmax_s <= 0: vmax_s = 1.0

    if mesh_info is not None:
        # Mesh-background path: decouple colour (global) and alpha
        # (per-frame) via manual RGBA so a small-amplitude frame in
        # `per-domain-global` mode stays visible instead of fading to
        # uniform transparent. imshow is valid here because GMT writer
        # emits a regular grid.
        vmax_f_alpha = _symmetric_vmax(Zf, percentile)
        vmax_s_alpha = _symmetric_vmax(Zs, percentile)
        ext_f = [float(Xf.min()), float(Xf.max()),
                 float(Yf.min()), float(Yf.max())]
        ext_s = [float(Xs.min()), float(Xs.max()),
                 float(Ys.min()), float(Ys.max())]
        ax.imshow(_wave_rgba(Zf, vmax_f, vmax_f_alpha,
                             cmap_name=cmap_name, mode=mode),
                  extent=ext_f, origin="lower", aspect="auto",
                  zorder=3, interpolation="nearest")
        ax.imshow(_wave_rgba(Zs, vmax_s, vmax_s_alpha,
                             cmap_name=cmap_name, mode=mode),
                  extent=ext_s, origin="lower", aspect="auto",
                  zorder=4, interpolation="nearest")
        # Dummy ScalarMappable for the colorbar (annotates the colour
        # scale, NOT the alpha scale — per-frame alpha is visual only).
        vmin_bar, vmax_bar = (0.0, 1.0) if mode == "magnitude" else (-1.0, 1.0)
        ticks = [0, 1] if mode == "magnitude" else [-1, 0, 1]
        sm = plt.cm.ScalarMappable(
            cmap=cmap_name,
            norm=mcolors.Normalize(vmin=vmin_bar, vmax=vmax_bar))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.8, ticks=ticks)
    else:
        # No mesh background: plain colormap, single-axis normalisation.
        effective_cmap = cmap if cmap is not None else cmap_name
        if mode == "magnitude":
            im_f = ax.pcolormesh(Xf, Yf, np.abs(Zf) / vmax_f,
                                 shading="auto", cmap=effective_cmap,
                                 vmin=0, vmax=1, zorder=3)
            ax.pcolormesh(Xs, Ys, np.abs(Zs) / vmax_s, shading="auto",
                          cmap=effective_cmap, vmin=0, vmax=1, zorder=3)
            cb = plt.colorbar(im_f, ax=ax, shrink=0.8, ticks=[0, 1])
        else:
            im_f = ax.pcolormesh(Xf, Yf, Zf / vmax_f, shading="auto",
                                 cmap=effective_cmap, vmin=-1, vmax=1,
                                 zorder=3)
            ax.pcolormesh(Xs, Ys, Zs / vmax_s, shading="auto",
                          cmap=effective_cmap, vmin=-1, vmax=1, zorder=3)
            cb = plt.colorbar(im_f, ax=ax, shrink=0.8, ticks=[-1, 0, 1])
    cb.set_label(
        f"normalised amplitude\n"
        f"{fluid_label}: ± {vmax_f:.2e}   |   "
        f"{solid_label}: ± {vmax_s:.2e}",
        fontsize=8,
    )


def plot_coupled_material(results_dir: Path, outdir: Path) -> None:
    """Plot fluid+solid material fields side by side. Each side is
    handled independently (like the pure path), just with a 'fluid_' /
    'solid_' prefix on the saved PNG."""
    any_plotted = False
    for side in ("fluid", "solid"):
        side_dir = results_dir / side
        gmt_dir = side_dir / "material" / "gmt"
        if not gmt_dir.is_dir():
            continue
        info_path = gmt_dir / "grid_info.txt"
        grid_info = read_grid_info(info_path) if info_path.exists() else None

        fields = {}
        for f in sorted(gmt_dir.glob("*.xyz")):
            X, Y, Z = load_xyz(f, grid_info)
            fields[f.stem] = (X, Y, Z)
        if not fields:
            continue

        nfields = len(fields)
        fig, axes = plt.subplots(1, nfields, figsize=(6 * nfields, 5))
        if nfields == 1:
            axes = [axes]
        for ax, (name, (X, Y, Z)) in zip(axes, fields.items()):
            constant, mean_val = is_constant_field(Z)
            if constant:
                margin = abs(mean_val) * 0.01 if mean_val != 0 else 1.0
                im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis",
                                   vmin=mean_val - margin,
                                   vmax=mean_val + margin)
                cb = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[mean_val])
                cb.ax.set_yticklabels([f"{mean_val:g}"])
            else:
                im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
                plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"{side}:{name}")
            ax.set_aspect("equal")
        plt.tight_layout()
        outfile = outdir / f"material_{side}.{_OUT_EXT}"
        plt.savefig(outfile, dpi=150)
        plt.close()
        print(f"  Material ({side}) -> {outfile}")
        any_plotted = True

    if not any_plotted:
        print("  No coupled material GMT directories found")


def _plot_one_coupled_field(fluid_snaps: dict,
                            solid_snaps: dict,
                            fluid_prefix: str,
                            solid_prefix: str,
                            outdir: Path,
                            vmax_f_fixed: float | None = None,
                            vmax_s_fixed: float | None = None,
                            mesh_info=None,
                            fluid_bg: str = "#b8dff5",
                            solid_bg: str = "#d0d0d0",
                            show_edges: bool = False,
                            cmap_name: str = "seismic",
                            mode: str = "signed",
                            percentile: float = 100.0) -> None:
    """Render one PNG (or paginated series) pairing the given fluid
    and solid field prefixes. Both sides use the same `seismic`
    colormap with per-domain independent normalisation so the two
    submesh regions stitch into one visually continuous wavefield."""
    steps = sorted(set(fluid_snaps) | set(solid_snaps))
    if not steps:
        return

    pages = split_evenly(len(steps), MAX_PANELS_PER_PAGE)
    offset = 0

    # Build a filename-safe field pair tag (e.g. "PS_vs_DISP_mag").
    tag_f = fluid_prefix or "none"
    tag_s = solid_prefix or "none"
    pair_tag = f"{tag_f}__vs__{tag_s}"

    for page_idx, page_count in enumerate(pages):
        page_steps = steps[offset:offset + page_count]
        offset += page_count

        ncols = min(4, len(page_steps))
        nrows = -(-len(page_steps) // ncols)
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(6 * ncols, 5 * nrows))
        if nrows * ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes[np.newaxis, :]
        elif ncols == 1:
            axes = axes[:, np.newaxis]

        for idx, step in enumerate(page_steps):
            r, c = idx // ncols, idx % ncols
            ax = axes[r, c]

            fluid = fluid_snaps.get(step)
            solid = solid_snaps.get(step)

            def _draw_single_side(X, Y, Z, vmax_fixed, label):
                _add_mesh_background(ax, mesh_info, fluid_bg, solid_bg,
                                     show_edges)
                vmax_color = (vmax_fixed if vmax_fixed is not None
                              else _symmetric_vmax(Z, percentile))
                if vmax_color <= 0: vmax_color = 1.0
                if mesh_info is not None:
                    vmax_alpha = _symmetric_vmax(Z, percentile)
                    ext = [float(X.min()), float(X.max()),
                           float(Y.min()), float(Y.max())]
                    ax.imshow(_wave_rgba(Z, vmax_color, vmax_alpha,
                                         cmap_name=cmap_name, mode=mode),
                              extent=ext, origin="lower", aspect="auto",
                              zorder=3, interpolation="nearest")
                    vmin_bar, vmax_bar = ((0.0, 1.0) if mode == "magnitude"
                                          else (-1.0, 1.0))
                    ticks = [0, 1] if mode == "magnitude" else [-1, 0, 1]
                    sm = plt.cm.ScalarMappable(
                        cmap=cmap_name,
                        norm=mcolors.Normalize(vmin=vmin_bar, vmax=vmax_bar))
                    sm.set_array([])
                    label_prefix = ("|·|" if mode == "magnitude" else "±")
                    plt.colorbar(sm, ax=ax, shrink=0.7, ticks=ticks,
                                 label=f"{label} ({label_prefix} {vmax_color:.2e})")
                else:
                    if mode == "magnitude":
                        im = ax.pcolormesh(X, Y, np.abs(Z) / vmax_color,
                                           shading="auto", cmap=cmap_name,
                                           vmin=0, vmax=1, zorder=3)
                        plt.colorbar(im, ax=ax, shrink=0.7,
                                     label=f"{label} (|·| {vmax_color:.2e})")
                    else:
                        im = ax.pcolormesh(X, Y, Z / vmax_color, shading="auto",
                                           cmap=cmap_name, vmin=-1, vmax=1,
                                           zorder=3)
                        plt.colorbar(im, ax=ax, shrink=0.7,
                                     label=f"{label} (± {vmax_color:.2e})")

            if fluid is not None and solid is not None:
                _overlay_two_domains(
                    ax, fluid, solid,
                    fluid_label=f"{fluid_prefix} [fluid]",
                    solid_label=f"{solid_prefix} [solid]",
                    vmax_f_fixed=vmax_f_fixed,
                    vmax_s_fixed=vmax_s_fixed,
                    mesh_info=mesh_info,
                    fluid_bg=fluid_bg, solid_bg=solid_bg,
                    show_edges=show_edges,
                    cmap_name=cmap_name, mode=mode,
                    percentile=percentile,
                )
            elif fluid is not None:
                X, Y, Z = fluid
                _draw_single_side(X, Y, Z, vmax_f_fixed,
                                   f"{fluid_prefix} [fluid]")
            elif solid is not None:
                X, Y, Z = solid
                _draw_single_side(X, Y, Z, vmax_s_fixed,
                                   f"{solid_prefix} [solid]")

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"Step {step}")
            ax.set_aspect("equal")

        for idx in range(len(page_steps), nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        base_title = (f"Coupled Wavefield "
                      f"(fluid {fluid_prefix or '—'} + "
                      f"solid {solid_prefix or '—'})")
        if len(pages) == 1:
            suptitle = base_title
            outfile = outdir / f"wavefield_coupled_{pair_tag}.{_OUT_EXT}"
        else:
            suptitle = f"{base_title} ({page_idx + 1}/{len(pages)})"
            outfile = (outdir /
                       f"wavefield_coupled_{pair_tag}_p{page_idx + 1}.{_OUT_EXT}")

        plt.suptitle(suptitle, fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Coupled wavefield -> {outfile}")


def _global_vmax(snaps: dict, percentile: float = 100.0) -> float:
    """Global (cross-timestep) |Z| upper bound for a snapshot dict.

    * `percentile == 100` (default): true max across every frame —
      backwards compatible.
    * `percentile < 100`: concatenate every frame's valid |Z|, then
      take that percentile. Useful when one source-peak frame would
      otherwise pull vmax so high that every other frame flattens
      into near-transparency."""
    if percentile >= 100.0 or percentile <= 0.0:
        vmax = 0.0
        for _, (_, _, Z) in snaps.items():
            valid = Z[~np.isnan(Z)]
            if len(valid):
                v = float(np.max(np.abs(valid)))
                if v > vmax:
                    vmax = v
        return vmax if vmax > 0 else 1.0
    # percentile path: aggregate all valid |Z| across frames, one pass.
    chunks = []
    for _, (_, _, Z) in snaps.items():
        valid = Z[~np.isnan(Z)]
        if len(valid):
            chunks.append(np.abs(valid))
    if not chunks:
        return 1.0
    v = float(np.percentile(np.concatenate(chunks), percentile))
    return v if v > 0 else 1.0


def plot_coupled_wavefield(results_dir: Path, outdir: Path,
                           normalize: str = "per-frame",
                           mesh_info=None,
                           fluid_bg: str = "#b8dff5",
                           solid_bg: str = "#d0d0d0",
                           show_edges: bool = False,
                           cmap_name: str = "seismic",
                           mode: str = "signed",
                           skip: int = 1,
                           percentile: float = 100.0) -> None:
    """Emit one PNG (or paginated series) per (fluid_prefix,
    solid_prefix) pair, covering every field actually written to the
    GMT subdirectories of both domains.

    Rationale: when the YAML enables multiple fields (e.g. fluid PS and
    DISP, solid DISP_mag and VEL_mag) the user expects to see all of
    them, not a single prefix chosen by internal priority. Each pair
    gets its own overlay panel series so axis colorbars stay meaningful
    per-field.

    `normalize` controls the symmetric vmax used for each domain:
      * `per-frame`       — Z / |Z|.max of the current snapshot only.
                            Best for seeing wavefronts as they decay.
      * `per-domain-global` — max |Z| across ALL timesteps for that
                            side. Constant per-domain vmax across
                            frames, so amplitude decay is visible
                            while the two domains stay independently
                            scaled. Recommended for coupled.
      * `global`          — single max across BOTH domains AND all
                            timesteps. Simplest to interpret, but the
                            Pa-vs-µm scale gap typically makes the
                            smaller-amplitude side invisible."""
    fluid_gmt = results_dir / "fluid" / "wavefield" / "gmt"
    solid_gmt = results_dir / "solid" / "wavefield" / "gmt"
    if not fluid_gmt.is_dir() and not solid_gmt.is_dir():
        print("  No coupled wavefield GMT directories found")
        return

    fluid_prefixes = (_list_field_prefixes(fluid_gmt)
                      if fluid_gmt.is_dir() else [])
    solid_prefixes = (_list_field_prefixes(solid_gmt)
                      if solid_gmt.is_dir() else [])

    if not fluid_prefixes and not solid_prefixes:
        print("  No coupled wavefield snapshots found")
        return

    # Fluid-only or solid-only runs: still emit a panel series for the
    # available side using an empty opposite prefix.
    if not fluid_prefixes:
        fluid_prefixes = [""]
    if not solid_prefixes:
        solid_prefixes = [""]

    print(f"  fluid prefixes: {fluid_prefixes}")
    print(f"  solid prefixes: {solid_prefixes}")
    print(f"  normalize mode: {normalize}")

    for fp in fluid_prefixes:
        fluid_snaps = (_load_snapshot_dict(fluid_gmt, fp)
                       if fp and fluid_gmt.is_dir() else {})
        for sp in solid_prefixes:
            solid_snaps = (_load_snapshot_dict(solid_gmt, sp)
                           if sp and solid_gmt.is_dir() else {})

            # Pre-compute per-pair vmax according to `normalize` mode.
            vmax_f_fixed = vmax_s_fixed = None
            if normalize == "per-domain-global":
                if fluid_snaps:
                    vmax_f_fixed = _global_vmax(fluid_snaps, percentile)
                if solid_snaps:
                    vmax_s_fixed = _global_vmax(solid_snaps, percentile)
            elif normalize == "global":
                vs = [_global_vmax(d, percentile)
                      for d in (fluid_snaps, solid_snaps) if d]
                if vs:
                    v = max(vs)
                    if fluid_snaps: vmax_f_fixed = v
                    if solid_snaps: vmax_s_fixed = v
            # else: per-frame (None → _overlay_two_domains recomputes each frame)

            # Apply `skip` stride AFTER the global vmax pass so that
            # constant-scale plots use the full-run peak even when the
            # rendered panels subsample.
            if skip > 1:
                def _stride_dict(d, s):
                    keys = sorted(d.keys())[::s]
                    return {k: d[k] for k in keys}
                fluid_snaps_plot = _stride_dict(fluid_snaps, skip)
                solid_snaps_plot = _stride_dict(solid_snaps, skip)
            else:
                fluid_snaps_plot = fluid_snaps
                solid_snaps_plot = solid_snaps

            _plot_one_coupled_field(fluid_snaps_plot, solid_snaps_plot,
                                    fp, sp, outdir,
                                    vmax_f_fixed=vmax_f_fixed,
                                    vmax_s_fixed=vmax_s_fixed,
                                    mesh_info=mesh_info,
                                    fluid_bg=fluid_bg,
                                    solid_bg=solid_bg,
                                    show_edges=show_edges,
                                    cmap_name=cmap_name, mode=mode,
                                    percentile=percentile)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Plot GMT xyz files")
    parser.add_argument("--results-dir", default="./results",
                        help="Results directory (default: ./results)")
    parser.add_argument("--coupled", action="store_true",
                        help="Force coupled-mode plotting (overlay fluid "
                             "and solid snapshots). Auto-detected when "
                             "<results>/fluid or <results>/solid contain "
                             "gmt/ subfolders.")
    parser.add_argument(
        "--normalize", default="per-frame",
        choices=["per-frame", "per-domain-global", "global"],
        help=("Wavefield normalisation strategy. "
              "`per-frame`: every panel uses its own |Z|.max (default). "
              "`per-domain-global`: (coupled only) max over all timesteps "
              "per domain (constant scale per side, amplitude decay "
              "visible, no cross-domain shadow). "
              "`global`: single max across all timesteps (and both "
              "domains in coupled); simplest, but the smaller-amplitude "
              "side typically disappears."))
    parser.add_argument(
        "--mesh", type=Path, default=None,
        help=("Path to the GMSH 2.2 mesh file. If given, fluid/solid "
              "element regions are painted under the wavefield with "
              "`--fluid-bg` / `--solid-bg`. Auto-detected from the "
              "config's `mesh.file` or the first *.msh sitting next "
              "to `--results-dir` when omitted."))
    parser.add_argument("--fluid-bg", default="#b8dff5",
                        help="Fluid region background colour.")
    parser.add_argument("--solid-bg", default="#d0d0d0",
                        help="Solid region background colour.")
    parser.add_argument("--mesh-edges", action="store_true",
                        help="Overlay thin mesh edges on top of the "
                             "background tints.")
    parser.add_argument("--no-mesh-bg", action="store_true",
                        help="Skip painting fluid/solid backgrounds "
                             "(e.g. when no mesh file is handy).")
    parser.add_argument(
        "--cmap", default=None,
        help=("Matplotlib colormap name. Defaults to `seismic` (signed) "
              "or `turbo` (magnitude). Use sequential cmaps like "
              "`turbo` / `inferno` / `magma` / `viridis` together "
              "with `--magnitude` for a single-sign energy view."))
    parser.add_argument(
        "--magnitude", action="store_true",
        help=("Plot |Z| with a sequential colormap instead of Z with "
              "a diverging one. Useful when the solid side only "
              "outputs `DISP_mag` (always >= 0); both sides then show "
              "matching non-negative intensities and wavefronts "
              "appear continuous across the fluid-solid interface."))
    parser.add_argument(
        "--skip", type=int, default=1,
        help=("Stride through snapshots (1 = every frame, 4 = every "
              "4th frame). Applied AFTER snapshot discovery, so the "
              "global vmax for `--normalize global`/`per-domain-"
              "global` still reflects the full run."))
    parser.add_argument(
        "--format", dest="out_format", default="png",
        choices=["png", "pdf"],
        help=("Output image format. `png` (default) = rasterised bitmap, "
              "`pdf` = vector output (good for publications; individual "
              "per-page files, not a combined PDF)."))
    parser.add_argument(
        "--percentile", type=float, default=100.0,
        help=("Clip vmax against this percentile of |Z| instead of the "
              "true max (seismic-imaging gain boost; stops a single "
              "source-spike pixel from compressing every other frame's "
              "dynamic range). 100 (default) = no clipping; 99.5 = "
              "gentle; 95 = aggressive. Applied to both per-frame and "
              "global vmax computations."))
    args = parser.parse_args()

    if args.skip < 1:
        args.skip = 1
    # Pick a sensible default cmap per mode if the user didn't set one.
    if args.cmap is None:
        args.cmap = "turbo" if args.magnitude else "seismic"
    plot_mode = "magnitude" if args.magnitude else "signed"
    # Flip every output path to the requested extension before any
    # plotting function constructs an outfile.
    global _OUT_EXT
    _OUT_EXT = args.out_format

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    outdir = results_dir / "figures"
    outdir.mkdir(exist_ok=True)

    use_coupled = args.coupled or is_coupled_layout(results_dir)

    # Resolve mesh path (CLI → auto-detect → None). Only matters for
    # coupled plots where we paint fluid/solid regions.
    mesh_info = None
    if use_coupled and not args.no_mesh_bg:
        mesh_path = args.mesh or _autodetect_mesh(results_dir)
        if mesh_path is not None and mesh_path.is_file():
            print(f"  mesh background from: {mesh_path}")
            mesh_info = _load_mesh_info(mesh_path)
        else:
            print("  mesh background: skipped (no mesh file found; "
                  "use --mesh to point at one, or --no-mesh-bg to "
                  "silence this)")

    if use_coupled:
        print("Plotting coupled material fields...")
        plot_coupled_material(results_dir, outdir)
        print("Plotting coupled wavefield snapshots...")
        plot_coupled_wavefield(results_dir, outdir,
                               normalize=args.normalize,
                               mesh_info=mesh_info,
                               fluid_bg=args.fluid_bg,
                               solid_bg=args.solid_bg,
                               show_edges=args.mesh_edges,
                               cmap_name=args.cmap, mode=plot_mode,
                               skip=args.skip,
                               percentile=args.percentile)
    else:
        # Pure single-domain has no separate per-domain axis → collapse
        # `per-domain-global` into `global`.
        pure_norm = "global" if args.normalize != "per-frame" else "per-frame"
        print("Plotting material fields...")
        plot_material(results_dir, outdir)
        print("Plotting wavefield snapshots...")
        plot_wavefield(results_dir, outdir, normalize=pure_norm,
                       cmap_name=args.cmap, mode=plot_mode,
                       skip=args.skip, percentile=args.percentile)

    print(f"Done. Figures saved to {outdir}/")


if __name__ == "__main__":
    main()
