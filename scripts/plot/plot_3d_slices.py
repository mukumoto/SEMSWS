#!/usr/bin/env python3
"""
Plot 3D cross-sections using PyVista for proper 3D rendering.

Places yz, xz, xy slices at their physical positions with correct
depth buffering and intersection handling.

Usage:
    python plot_3d_slices.py [--results-dir ./results] [--type wavefield] [--step 600]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
try:
    pv.start_xvfb()
except Exception:
    pass


def read_grid_info(info_path):
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
    data = np.loadtxt(filepath)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]
    if grid_info:
        nx, ny = int(grid_info["nx"]), int(grid_info["ny"])
    else:
        nx = 1
        for i in range(1, len(x)):
            if x[i] <= x[i - 1]:
                nx = i
                break
        ny = len(x) // nx
    return x.reshape(ny, nx), y.reshape(ny, nx), z.reshape(ny, nx)


def parse_slice_name(name):
    """Parse 'yz_x12000' -> ('yz', 'x', 12000.0)"""
    plane = name[:2]
    rest = name[3:]
    fixed_axis = rest[0]
    fixed_val = float(rest[1:])
    return plane, fixed_axis, fixed_val


def make_structured_grid(plane, fixed_val, X2d, Y2d, values):
    """Create a PyVista StructuredGrid for a cross-section.

    X2d, Y2d, values are (ny, nx) arrays from load_xyz.
    PyVista StructuredGrid uses Fortran (column-major) point ordering,
    so we transpose to (nx, ny) to align correctly.
    """
    # Transpose from (ny, nx) to (nx, ny) for PyVista's Fortran ordering
    X2d = X2d.T
    Y2d = Y2d.T
    values = values.T

    if plane == "yz":
        # fixed x, col1=y, col2=z
        x3d = np.full_like(X2d, fixed_val)
        y3d = X2d
        z3d = Y2d
    elif plane == "xz":
        # fixed y, col1=x, col2=z
        x3d = X2d
        y3d = np.full_like(X2d, fixed_val)
        z3d = Y2d
    elif plane == "xy":
        # fixed z, col1=x, col2=y
        x3d = X2d
        y3d = Y2d
        z3d = np.full_like(X2d, fixed_val)
    else:
        raise ValueError(f"Unknown plane: {plane}")

    grid = pv.StructuredGrid(x3d, y3d, z3d)
    grid.point_data["values"] = values.ravel(order="F")
    return grid


def plot_3d_assembly(gmt_dir, field_pattern, output_file, title,
                     symmetric=False):
    """Load cross-sections and render them in 3D with PyVista."""
    slice_dirs = sorted([d for d in gmt_dir.iterdir() if d.is_dir()])
    if not slice_dirs:
        print(f"  No cross-section directories in {gmt_dir}")
        return

    # Load slices
    grids = []
    for sd in slice_dirs:
        info_path = sd / "grid_info.txt"
        grid_info = read_grid_info(info_path) if info_path.exists() else None

        xyz_files = sorted(sd.glob(field_pattern))
        if not xyz_files:
            continue

        f = xyz_files[0]
        X, Y, Z = load_xyz(f, grid_info)
        plane, fixed_axis, fixed_val = parse_slice_name(sd.name)
        grid = make_structured_grid(plane, fixed_val, X, Y, Z)
        grids.append(grid)

    if not grids:
        print(f"  No matching files for '{field_pattern}'")
        return

    # Color range
    all_vals = np.concatenate([g.point_data["values"] for g in grids])
    all_vals = all_vals[~np.isnan(all_vals)]

    if symmetric:
        vmax = np.max(np.abs(all_vals)) if len(all_vals) > 0 else 1.0
        if vmax == 0:
            vmax = 1.0
        clim = [-vmax, vmax]
        cmap = "seismic"
    else:
        vmin, vmax = np.min(all_vals), np.max(all_vals)
        if vmin == vmax:
            vmin -= 1
            vmax += 1
        clim = [vmin, vmax]
        cmap = "viridis"

    # Render
    plotter = pv.Plotter(off_screen=True, window_size=[1600, 1200])
    plotter.set_background("white")

    for i, grid in enumerate(grids):
        # Replace NaN with 0 so the full surface is visible
        vals = grid.point_data["values"].copy()
        vals[np.isnan(vals)] = 0.0
        grid.point_data["values"] = vals

        plotter.add_mesh(
            grid,
            scalars="values",
            cmap=cmap,
            clim=clim,
            show_scalar_bar=False,
            opacity=1.0,
            lighting=False,
        )

    # Bounding box with grid lines and axis labels
    try:
        plotter.show_bounds(
            grid="back",
            location="outer",
            ticks="both",
            xtitle="X (m)",
            ytitle="Y (m)",
            ztitle="Z (m)",
            font_size=12,
            color="black",
        )
    except TypeError:
        # Older PyVista versions don't support xtitle/ytitle/ztitle
        plotter.show_bounds(
            grid="back",
            location="outer",
            ticks="both",
            font_size=12,
            color="black",
        )

    # Scalar bar at bottom
    plotter.add_scalar_bar(
        title=title,
        position_x=0.2,
        position_y=0.03,
        width=0.6,
        height=0.06,
        label_font_size=14,
        title_font_size=16,
        color="black",
        n_labels=5,
    )

    # Camera: 45 deg on xy plane, moderate elevation, pulled back
    plotter.reset_camera()
    plotter.camera.azimuth = -80
    plotter.camera.elevation = 10
    plotter.camera.zoom(0.8)

    plotter.screenshot(str(output_file))
    plotter.close()
    print(f"  -> {output_file}")


# --------------------------------------------------------------------------
# Coupled fluid-solid 3D slice rendering
# --------------------------------------------------------------------------


def is_coupled_layout(results_dir: Path) -> bool:
    """A coupled 3D run writes its GMT slices under <results>/fluid/
    and <results>/solid/. Either side having a gmt/ subtree qualifies."""
    for side in ("fluid", "solid"):
        sd = results_dir / side
        if (sd / "wavefield" / "gmt").is_dir() or (sd / "material" / "gmt").is_dir():
            return True
    return False


def _list_prefixes(slice_dir: Path) -> list[str]:
    """Enumerate `<prefix>_NNNNNN.xyz` prefixes present in a slice dir."""
    prefixes: set[str] = set()
    for f in slice_dir.glob("*_[0-9][0-9][0-9][0-9][0-9][0-9].xyz"):
        stem = f.stem
        if len(stem) > 7 and stem[-7] == "_" and stem[-6:].isdigit():
            prefixes.add(stem[:-7])
    return sorted(prefixes)


def _plot_coupled_3d_wavefield_step(
    fluid_wf: Path, solid_wf: Path, slices: list[str],
    fluid_prefix: str, solid_prefix: str, step_str: str,
    output_file: Path, title: str):
    """Render one 3D frame pairing fluid + solid slices at `step_str`.

    Each domain is normalised independently to ±1 and drawn with the
    same `seismic` colormap, so the two sides stitch into one visually
    connected 3D region (NaN outside each submesh is replaced with
    zero, which reads as the neutral mid-colormap colour)."""
    fluid_grids, solid_grids = [], []
    vmax_f = vmax_s = None

    for slice_name in slices:
        plane, fixed_axis, fixed_val = parse_slice_name(slice_name)

        fluid_slice = fluid_wf / slice_name
        solid_slice = solid_wf / slice_name

        if fluid_prefix and fluid_slice.is_dir():
            f = fluid_slice / f"{fluid_prefix}_{step_str}.xyz"
            if f.exists():
                info = fluid_slice / "grid_info.txt"
                grid_info = read_grid_info(info) if info.exists() else None
                X, Y, Z = load_xyz(f, grid_info)
                grid = make_structured_grid(plane, fixed_val, X, Y, Z)
                fluid_grids.append(grid)
                vals = Z[~np.isnan(Z)]
                if len(vals) > 0:
                    m = float(np.max(np.abs(vals)))
                    vmax_f = max(vmax_f or 0.0, m)

        if solid_prefix and solid_slice.is_dir():
            f = solid_slice / f"{solid_prefix}_{step_str}.xyz"
            if f.exists():
                info = solid_slice / "grid_info.txt"
                grid_info = read_grid_info(info) if info.exists() else None
                X, Y, Z = load_xyz(f, grid_info)
                grid = make_structured_grid(plane, fixed_val, X, Y, Z)
                solid_grids.append(grid)
                vals = Z[~np.isnan(Z)]
                if len(vals) > 0:
                    m = float(np.max(np.abs(vals)))
                    vmax_s = max(vmax_s or 0.0, m)

    if not fluid_grids and not solid_grids:
        return

    # Normalise in place and replace NaN with 0 (neutral mid-colormap).
    for g, vmax in ((fluid_grids, vmax_f or 1.0),
                    (solid_grids, vmax_s or 1.0)):
        for gr in g:
            v = gr.point_data["values"].copy()
            v[np.isnan(v)] = 0.0
            gr.point_data["values"] = v / vmax

    plotter = pv.Plotter(off_screen=True, window_size=[1600, 1200])
    plotter.set_background("white")
    for gr in fluid_grids + solid_grids:
        plotter.add_mesh(gr, scalars="values", cmap="seismic",
                         clim=[-1.0, 1.0], show_scalar_bar=False,
                         opacity=1.0, lighting=False)

    try:
        plotter.show_bounds(grid="back", location="outer", ticks="both",
                            xtitle="X (m)", ytitle="Y (m)", ztitle="Z (m)",
                            font_size=12, color="black")
    except TypeError:
        plotter.show_bounds(grid="back", location="outer", ticks="both",
                            font_size=12, color="black")

    bar_parts = []
    if vmax_f is not None:
        bar_parts.append(f"{fluid_prefix} [fluid] ± {vmax_f:.2e}")
    if vmax_s is not None:
        bar_parts.append(f"{solid_prefix} [solid] ± {vmax_s:.2e}")
    plotter.add_scalar_bar(title=" | ".join(bar_parts),
                           position_x=0.15, position_y=0.03,
                           width=0.7, height=0.06,
                           label_font_size=12, title_font_size=14,
                           color="black", n_labels=5)

    plotter.reset_camera()
    plotter.camera.azimuth = -80
    plotter.camera.elevation = 10
    plotter.camera.zoom(0.8)

    plotter.screenshot(str(output_file))
    plotter.close()
    print(f"  -> {output_file}")


def plot_coupled_3d(results_dir: Path, outdir: Path,
                    plot_material: bool, plot_wavefield: bool,
                    step_filter: int | None):
    """Coupled 3D rendering: one PNG per (fluid_prefix × solid_prefix)
    pair per step, plus per-side material assemblies."""
    # Materials: render each side independently (the single-domain path
    # already produces meaningful 3D assemblies per field).
    if plot_material:
        for side in ("fluid", "solid"):
            mat_gmt = results_dir / side / "material" / "gmt"
            if not mat_gmt.exists():
                continue
            print(f"Plotting coupled material 3D assembly ({side})...")
            first_slice = next(
                (d for d in sorted(mat_gmt.iterdir()) if d.is_dir()), None)
            if first_slice:
                for f in sorted(first_slice.glob("*.xyz")):
                    field = f.stem
                    plot_3d_assembly(
                        mat_gmt, f"{field}.xyz",
                        outdir / f"3d_material_{side}_{field}.png",
                        title=f"{side}:{field}")

    if not plot_wavefield:
        return

    fluid_wf = results_dir / "fluid" / "wavefield" / "gmt"
    solid_wf = results_dir / "solid" / "wavefield" / "gmt"
    if not fluid_wf.is_dir() and not solid_wf.is_dir():
        return

    # Slice set = union of per-side slice subdirs.
    slices: set[str] = set()
    if fluid_wf.is_dir():
        slices.update(d.name for d in fluid_wf.iterdir() if d.is_dir())
    if solid_wf.is_dir():
        slices.update(d.name for d in solid_wf.iterdir() if d.is_dir())
    if not slices:
        print("  No coupled wavefield slices found")
        return
    slices_sorted = sorted(slices)

    # Pick any valid slice on each side to enumerate prefixes.
    fluid_prefixes: list[str] = []
    solid_prefixes: list[str] = []
    for s in slices_sorted:
        if fluid_wf.is_dir() and (fluid_wf / s).is_dir() and not fluid_prefixes:
            fluid_prefixes = _list_prefixes(fluid_wf / s)
        if solid_wf.is_dir() and (solid_wf / s).is_dir() and not solid_prefixes:
            solid_prefixes = _list_prefixes(solid_wf / s)
    if not fluid_prefixes:
        fluid_prefixes = [""]
    if not solid_prefixes:
        solid_prefixes = [""]

    # Collect the step set as the intersection/union of files across
    # any slice directory (all slices are written in lockstep).
    def collect_steps(slice_dir: Path, prefix: str) -> list[str]:
        if not prefix or not slice_dir.is_dir():
            return []
        return sorted(
            f.stem[len(prefix) + 1:]
            for f in slice_dir.glob(f"{prefix}_[0-9]*.xyz")
        )

    print("Plotting coupled wavefield 3D assembly...")
    for fp in fluid_prefixes:
        for sp in solid_prefixes:
            # Steps: union across fluid and solid for the first slice
            # that has this prefix (all slices share step cadence).
            ref_slice = next(
                (s for s in slices_sorted
                 if (fluid_wf / s).is_dir() or (solid_wf / s).is_dir()),
                None)
            steps: list[str] = []
            if ref_slice:
                steps_f = collect_steps(fluid_wf / ref_slice, fp)
                steps_s = collect_steps(solid_wf / ref_slice, sp)
                steps = sorted(set(steps_f) | set(steps_s))
            if step_filter is not None:
                target = f"{step_filter:06d}"
                steps = [target] if target in steps else []

            pair_tag = f"{fp or 'none'}__vs__{sp or 'none'}"
            for step_str in steps:
                outfile = (outdir /
                           f"3d_coupled_{pair_tag}_{step_str}.png")
                _plot_coupled_3d_wavefield_step(
                    fluid_wf, solid_wf, slices_sorted,
                    fp, sp, step_str, outfile,
                    title=f"{fp or '—'} + {sp or '—'} (step {step_str})")


def main():
    parser = argparse.ArgumentParser(description="Plot 3D cross-sections with PyVista")
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--type", choices=["wavefield", "material", "both"],
                        default="both")
    parser.add_argument("--step", type=int, default=None,
                        help="Specific step for wavefield (default: all)")
    parser.add_argument("--coupled", action="store_true",
                        help="Force coupled-mode rendering (overlay fluid "
                             "+ solid slices). Auto-detected when "
                             "<results>/fluid or <results>/solid contain "
                             "gmt/ subtrees.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    outdir = results_dir / "figures"
    outdir.mkdir(exist_ok=True)

    if args.coupled or is_coupled_layout(results_dir):
        plot_coupled_3d(
            results_dir, outdir,
            plot_material=(args.type in ("material", "both")),
            plot_wavefield=(args.type in ("wavefield", "both")),
            step_filter=args.step,
        )
        print(f"Done. Figures saved to {outdir}/")
        return

    # Material
    if args.type in ("material", "both"):
        mat_gmt = results_dir / "material" / "gmt"
        if mat_gmt.exists():
            print("Plotting material 3D assembly...")
            # Auto-detect material fields from first slice directory
            first_slice = next(
                (d for d in sorted(mat_gmt.iterdir()) if d.is_dir()), None)
            if first_slice:
                for f in sorted(first_slice.glob("*.xyz")):
                    field = f.stem
                    plot_3d_assembly(
                        mat_gmt, f"{field}.xyz",
                        outdir / f"3d_{field}.png",
                        title=field)

    # Wavefield
    if args.type in ("wavefield", "both"):
        wf_gmt = results_dir / "wavefield" / "gmt"
        if wf_gmt.exists():
            print("Plotting wavefield 3D assembly...")
            # Auto-detect all field/component combinations from first slice dir
            first_slice = next(
                (d for d in sorted(wf_gmt.iterdir()) if d.is_dir()), None)
            if first_slice:
                # Group files by field+component prefix (e.g., DISP_mag, DISP_x, PS)
                all_files = sorted(first_slice.glob("*.xyz"))
                prefixes = set()
                for f in all_files:
                    # Parse: FIELD_comp_STEP.xyz or FIELD_STEP.xyz
                    parts = f.stem.rsplit("_", 1)  # split off step number
                    if len(parts) == 2:
                        prefixes.add(parts[0])

                for prefix in sorted(prefixes):
                    files = sorted(first_slice.glob(f"{prefix}_*.xyz"))
                    if args.step is not None:
                        pattern = f"{prefix}_{args.step:06d}.xyz"
                        plot_3d_assembly(
                            wf_gmt, pattern,
                            outdir / f"3d_{prefix}_{args.step:06d}.png",
                            title=f"{prefix} (step {args.step})",
                            symmetric=True)
                    else:
                        for f in files:
                            step_str = f.stem.split("_")[-1]
                            pattern = f"{prefix}_{step_str}.xyz"
                            plot_3d_assembly(
                                wf_gmt, pattern,
                                outdir / f"3d_{prefix}_{step_str}.png",
                                title=f"{prefix} (step {step_str})",
                                symmetric=True)

    print(f"Done. Figures saved to {outdir}/")


if __name__ == "__main__":
    main()
