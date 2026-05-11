#!/usr/bin/env python3
"""
Plot GMT xyz cross-section files for 3D acoustic visualization test.

Reads grid_info.txt for exact grid parameters.
Handles constant fields with appropriate colorbar.

Usage:
    python plot_wavefield.py [--results-dir ./results]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    """Load GMT xyz file using grid_info for exact reshape."""
    data = np.loadtxt(filepath)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    if grid_info:
        nx = int(grid_info["nx"])
        ny = int(grid_info["ny"])
    else:
        nx = 1
        for i in range(1, len(x)):
            if x[i] <= x[i - 1]:
                nx = i
                break
        ny = len(x) // nx

    return x.reshape(ny, nx), y.reshape(ny, nx), z.reshape(ny, nx)


def is_constant_field(Z, rtol=1e-4):
    """Check if field is approximately constant."""
    valid = Z[~np.isnan(Z)]
    if len(valid) == 0:
        return True, 0.0
    mean_val = np.mean(valid)
    if mean_val == 0:
        return np.all(valid == 0), 0.0
    rel_range = (np.max(valid) - np.min(valid)) / abs(mean_val)
    return rel_range < rtol, mean_val


def get_axis_labels(slice_name):
    """Extract axis labels from slice directory name (e.g., 'yz_x12000')."""
    plane = slice_name[:2]
    return plane[0].upper(), plane[1].upper()


MAX_PANELS_PER_PAGE = 12  # max panels per PNG (3 cols x 4 rows)


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


def plot_snapshot_page(snapshots, ax1_label, ax2_label, suptitle, outfile):
    """Plot a single page of wavefield snapshots."""
    ncols = min(3, len(snapshots))
    nrows = -(-len(snapshots) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows * ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes[np.newaxis, :]
    elif ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, (step_str, X, Y, Z) in enumerate(snapshots):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        valid = Z[~np.isnan(Z)]
        vmax = np.max(np.abs(valid)) if len(valid) > 0 else 1.0
        if vmax == 0:
            vmax = 1.0

        im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="seismic",
                           vmin=-vmax, vmax=vmax)
        ax.set_xlabel(f"{ax1_label} (m)")
        ax.set_ylabel(f"{ax2_label} (m)")
        ax.set_title(f"Step {step_str}")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.7)

    for idx in range(len(snapshots), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cross_sections(gmt_dir, output_dir, title_prefix, is_wavefield=False):
    """Plot all cross-sections in a GMT directory."""
    slice_dirs = sorted([d for d in gmt_dir.iterdir() if d.is_dir()])

    if not slice_dirs:
        print(f"  No cross-section directories found in {gmt_dir}")
        return

    for slice_dir in slice_dirs:
        info_path = slice_dir / "grid_info.txt"
        grid_info = read_grid_info(info_path) if info_path.exists() else None

        xyz_files = sorted(slice_dir.glob("*.xyz"))
        if not xyz_files:
            continue

        slice_name = slice_dir.name
        ax1_label, ax2_label = get_axis_labels(slice_name)
        print(f"  {slice_name}: {len(xyz_files)} files")

        out_slice_dir = output_dir / slice_name
        out_slice_dir.mkdir(parents=True, exist_ok=True)

        if is_wavefield:
            # Group files by prefix (e.g., DISP_mag, DISP_x, PS)
            from collections import defaultdict
            groups = defaultdict(list)
            for f in xyz_files:
                # Split: FIELD_comp_STEP.xyz or FIELD_STEP.xyz
                parts = f.stem.rsplit("_", 1)  # [prefix, step]
                if len(parts) == 2:
                    groups[parts[0]].append(f)

            for prefix in sorted(groups.keys()):
                files = sorted(groups[prefix])
                snapshots = []
                for f in files:
                    X, Y, Z = load_xyz(f, grid_info)
                    step_str = f.stem.split("_")[-1]
                    snapshots.append((step_str, X, Y, Z))

                pages = split_evenly(len(snapshots), MAX_PANELS_PER_PAGE)
                offset = 0
                for page_idx, page_count in enumerate(pages):
                    page_snapshots = snapshots[offset:offset + page_count]
                    offset += page_count

                    if len(pages) == 1:
                        suptitle = f"{prefix} - {slice_name}"
                        outfile = out_slice_dir / f"{prefix}.png"
                    else:
                        suptitle = f"{prefix} - {slice_name} ({page_idx + 1}/{len(pages)})"
                        outfile = out_slice_dir / f"{prefix}_{page_idx + 1}.png"

                    plot_snapshot_page(page_snapshots, ax1_label, ax2_label,
                                       suptitle, outfile)
                    print(f"    {prefix} -> {outfile}")

        else:
            # Material: one figure per field
            for f in xyz_files:
                X, Y, Z = load_xyz(f, grid_info)
                constant, mean_val = is_constant_field(Z)

                fig, ax = plt.subplots(1, 1, figsize=(8, 7))

                if constant:
                    margin = abs(mean_val) * 0.01 if mean_val != 0 else 1.0
                    im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis",
                                       vmin=mean_val - margin, vmax=mean_val + margin)
                    cb = plt.colorbar(im, ax=ax, shrink=0.8, ticks=[mean_val])
                    cb.ax.set_yticklabels([f"{mean_val:g}"])
                else:
                    im = ax.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
                    plt.colorbar(im, ax=ax, shrink=0.8)

                ax.set_xlabel(f"{ax1_label} (m)")
                ax.set_ylabel(f"{ax2_label} (m)")
                ax.set_title(f"{f.stem} ({slice_name})")
                ax.set_aspect("equal")

                plt.tight_layout()
                outfile = out_slice_dir / f"{f.stem}.png"
                plt.savefig(outfile, dpi=150)
                plt.close()
                print(f"    {f.stem} -> {outfile}")


# --------------------------------------------------------------------------
# Coupled fluid-solid cross-section plotting (3D)
# --------------------------------------------------------------------------


def is_coupled_layout(results_dir: Path) -> bool:
    """A coupled 3D run writes its GMT slices under <results>/fluid and
    <results>/solid. Either directory having a gmt/ tree qualifies."""
    for side in ("fluid", "solid"):
        sd = results_dir / side
        if (sd / "wavefield" / "gmt").is_dir() or (sd / "material" / "gmt").is_dir():
            return True
    return False


def _symmetric_vmax(Z: np.ndarray) -> float:
    valid = Z[~np.isnan(Z)]
    if len(valid) == 0:
        return 1.0
    v = float(np.max(np.abs(valid)))
    return v if v > 0 else 1.0


def _list_field_prefixes(slice_dir: Path) -> list[str]:
    """Enumerate every field prefix present in a 3D slice directory.

    Filenames are `<prefix>_NNNNNN.xyz` (6-digit zero-padded step).
    Returns the prefix list sorted for deterministic plot ordering."""
    prefixes: set[str] = set()
    for f in slice_dir.glob("*_[0-9][0-9][0-9][0-9][0-9][0-9].xyz"):
        stem = f.stem
        if len(stem) > 7 and stem[-7] == "_" and stem[-6:].isdigit():
            prefixes.add(stem[:-7])
    return sorted(prefixes)


def _load_snapshot_dict(slice_dir: Path, prefix: str) -> dict[str, tuple]:
    info_path = slice_dir / "grid_info.txt"
    grid_info = read_grid_info(info_path) if info_path.exists() else None
    out: dict[str, tuple] = {}
    for f in sorted(slice_dir.glob(f"{prefix}_[0-9]*.xyz")):
        step = f.stem[len(prefix) + 1:]
        X, Y, Z = load_xyz(f, grid_info)
        out[step] = (X, Y, Z)
    return out


def plot_coupled_cross_sections(results_dir: Path, outdir: Path) -> None:
    """Overlay fluid PS and solid DISP (or similar) on each slice of
    the 3D cross-section GMT output. Materials are rendered per side
    (like the pure path) into `<outdir>/material_{fluid,solid}/<slice>/`.
    Wavefields are rendered as overlays into `<outdir>/wavefield/<slice>/`
    because the point of the overlay is to show fluid and solid on the
    same slice at the same time."""
    # ----- Materials (one figure per side, per field, per slice) -----
    for side in ("fluid", "solid"):
        mat_gmt = results_dir / side / "material" / "gmt"
        if mat_gmt.exists():
            print(f"Plotting coupled material cross-sections ({side})...")
            plot_cross_sections(mat_gmt, outdir / f"material_{side}",
                                f"Material[{side}]", is_wavefield=False)

    # ----- Wavefield overlay -----
    fluid_wf = results_dir / "fluid" / "wavefield" / "gmt"
    solid_wf = results_dir / "solid" / "wavefield" / "gmt"
    if not fluid_wf.is_dir() and not solid_wf.is_dir():
        return

    # Slice set = union of per-side slice subdirs. Both sides should
    # have the same slices in practice (same parent grid), but be
    # tolerant to single-side-only layouts.
    slices: set[str] = set()
    if fluid_wf.is_dir():
        slices.update(d.name for d in fluid_wf.iterdir() if d.is_dir())
    if solid_wf.is_dir():
        slices.update(d.name for d in solid_wf.iterdir() if d.is_dir())
    if not slices:
        print("  No coupled wavefield slices found")
        return

    print("Plotting coupled wavefield cross-sections...")
    for slice_name in sorted(slices):
        fluid_slice = fluid_wf / slice_name if fluid_wf.is_dir() else None
        solid_slice = solid_wf / slice_name if solid_wf.is_dir() else None
        ax1_label, ax2_label = get_axis_labels(slice_name)

        fluid_prefixes = (_list_field_prefixes(fluid_slice)
                          if fluid_slice and fluid_slice.is_dir() else [])
        solid_prefixes = (_list_field_prefixes(solid_slice)
                          if solid_slice and solid_slice.is_dir() else [])
        if not fluid_prefixes and not solid_prefixes:
            continue
        if not fluid_prefixes:
            fluid_prefixes = [""]
        if not solid_prefixes:
            solid_prefixes = [""]

        out_slice_dir = outdir / "wavefield_coupled" / slice_name
        out_slice_dir.mkdir(parents=True, exist_ok=True)

        for fp in fluid_prefixes:
            fluid_snaps = (_load_snapshot_dict(fluid_slice, fp)
                           if fp and fluid_slice and fluid_slice.is_dir()
                           else {})
            for sp in solid_prefixes:
                solid_snaps = (_load_snapshot_dict(solid_slice, sp)
                               if sp and solid_slice and solid_slice.is_dir()
                               else {})
                steps = sorted(set(fluid_snaps) | set(solid_snaps))
                if not steps:
                    continue

                pair_tag = f"{fp or 'none'}__vs__{sp or 'none'}"
                pages = split_evenly(len(steps), MAX_PANELS_PER_PAGE)
                offset = 0
                for page_idx, page_count in enumerate(pages):
                    page_steps = steps[offset:offset + page_count]
                    offset += page_count

                    ncols = min(3, len(page_steps))
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

                        # Per-domain independent normalisation to ±1,
                        # shared `seismic` colormap so the two submesh
                        # grids stitch into one visually connected
                        # region on the same axes (each is NaN outside
                        # its own submesh).
                        im_handle = None
                        vmax_f = vmax_s = None
                        if fluid is not None:
                            Xf, Yf, Zf = fluid
                            vmax_f = _symmetric_vmax(Zf)
                            im_handle = ax.pcolormesh(
                                Xf, Yf, Zf / vmax_f,
                                shading="auto", cmap="seismic",
                                vmin=-1, vmax=1)
                        if solid is not None:
                            Xs, Ys, Zs = solid
                            vmax_s = _symmetric_vmax(Zs)
                            im_s = ax.pcolormesh(
                                Xs, Ys, Zs / vmax_s,
                                shading="auto", cmap="seismic",
                                vmin=-1, vmax=1)
                            im_handle = im_handle or im_s
                        if im_handle is not None:
                            cb = plt.colorbar(im_handle, ax=ax,
                                              shrink=0.7,
                                              ticks=[-1, 0, 1])
                            parts = []
                            if vmax_f is not None:
                                parts.append(f"{fp} [fluid] "
                                             f"± {vmax_f:.2e}")
                            if vmax_s is not None:
                                parts.append(f"{sp} [solid] "
                                             f"± {vmax_s:.2e}")
                            cb.set_label("normalised amplitude\n"
                                         + "   |   ".join(parts),
                                         fontsize=8)

                        ax.set_xlabel(f"{ax1_label} (m)")
                        ax.set_ylabel(f"{ax2_label} (m)")
                        ax.set_title(f"Step {step}")
                        ax.set_aspect("equal")

                    for idx in range(len(page_steps), nrows * ncols):
                        axes[idx // ncols, idx % ncols].set_visible(False)

                    base = (f"Coupled Wavefield — {slice_name} "
                            f"(fluid {fp or '—'} + solid {sp or '—'})")
                    if len(pages) == 1:
                        suptitle = base
                        outfile = out_slice_dir / f"{pair_tag}.png"
                    else:
                        suptitle = (f"{base} ({page_idx + 1}/{len(pages)})")
                        outfile = (out_slice_dir /
                                   f"{pair_tag}_p{page_idx + 1}.png")

                    plt.suptitle(suptitle, fontsize=14)
                    plt.tight_layout()
                    plt.savefig(outfile, dpi=150, bbox_inches="tight")
                    plt.close()
                    print(f"  {slice_name} {pair_tag} -> {outfile}")


def main():
    parser = argparse.ArgumentParser(description="Plot 3D GMT cross-section files")
    parser.add_argument("--results-dir", default="./results",
                        help="Results directory (default: ./results)")
    parser.add_argument("--coupled", action="store_true",
                        help="Force coupled-mode plotting (overlay fluid "
                             "and solid cross-sections per slice). "
                             "Auto-detected when <results>/fluid or "
                             "<results>/solid contain gmt/ subtrees.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    outdir = results_dir / "figures"
    outdir.mkdir(exist_ok=True)

    if args.coupled or is_coupled_layout(results_dir):
        plot_coupled_cross_sections(results_dir, outdir)
    else:
        # Material
        mat_gmt = results_dir / "material" / "gmt"
        if mat_gmt.exists():
            print("Plotting material cross-sections...")
            plot_cross_sections(mat_gmt, outdir / "material",
                                "Material", is_wavefield=False)

        # Wavefield
        wf_gmt = results_dir / "wavefield" / "gmt"
        if wf_gmt.exists():
            print("Plotting wavefield cross-sections...")
            plot_cross_sections(wf_gmt, outdir / "wavefield",
                                "Pressure", is_wavefield=True)

    print(f"Done. Figures saved to {outdir}/")


if __name__ == "__main__":
    main()
