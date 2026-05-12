"""
Generate HOHQMesh control file for "icm" text mesh.

Domain: 6000m x 3000m
Letters "icm" (without dot on i) as inner boundaries (holes).
Source will be placed at the "i dot" position.
Output: ABAQUS format for conversion to MFEM via meshio.
"""
import numpy as np
from matplotlib.textpath import TextToPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter1d
import argparse


def get_letter_contours(text, font_size=1.0, font_family="serif"):
    """Extract contours from text using matplotlib's font rendering."""
    fp = FontProperties(family=font_family, weight="bold", size=font_size)
    ttp = TextToPath()
    verts, codes = ttp.get_text_path(fp, text)
    path = Path(verts, codes)

    # Split into individual contours (each MOVETO starts a new contour)
    contours = []
    current = []
    for vert, code in zip(verts, codes):
        if code == Path.MOVETO:
            if len(current) > 2:
                contours.append(np.array(current))
            current = [vert]
        elif code == Path.CLOSEPOLY:
            if len(current) > 2:
                current.append(current[0])  # close
                contours.append(np.array(current))
            current = []
        elif code in (Path.LINETO, Path.CURVE3, Path.CURVE4):
            current.append(vert)
    if len(current) > 2:
        contours.append(np.array(current))

    return contours


def smooth_contour(contour, sigma=2.0):
    """Smooth a closed contour using Gaussian filter with periodic wrapping."""
    # Remove closing point, smooth, then re-close
    pts = contour[:-1]
    n = len(pts)
    # Pad periodically for smooth wrapping
    pad = min(n // 2, int(sigma * 5))
    x_padded = np.concatenate([pts[-pad:, 0], pts[:, 0], pts[:pad, 0]])
    y_padded = np.concatenate([pts[-pad:, 1], pts[:, 1], pts[:pad, 1]])
    x_smooth = gaussian_filter1d(x_padded, sigma=sigma)[pad:pad + n]
    y_smooth = gaussian_filter1d(y_padded, sigma=sigma)[pad:pad + n]
    smoothed = np.column_stack([x_smooth, y_smooth])
    return np.vstack([smoothed, smoothed[0]])


def resample_contour(contour, n_points=100):
    """Resample contour to have uniform spacing with n_points."""
    # Compute cumulative arc length
    diffs = np.diff(contour, axis=0)
    seg_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    cum_length = np.concatenate([[0], np.cumsum(seg_lengths)])
    total_length = cum_length[-1]

    if total_length < 1e-10:
        return None

    # Resample at uniform arc length
    t_uniform = np.linspace(0, total_length, n_points, endpoint=False)
    x_new = np.interp(t_uniform, cum_length, contour[:, 0])
    y_new = np.interp(t_uniform, cum_length, contour[:, 1])

    # Close the contour
    resampled = np.column_stack([x_new, y_new])
    resampled = np.vstack([resampled, resampled[0]])
    return resampled


def ensure_ccw(contour):
    """Ensure contour is counter-clockwise (required by HOHQMesh for inner boundaries)."""
    # Shoelace formula for signed area
    x, y = contour[:-1, 0], contour[:-1, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area > 0:  # clockwise
        return contour[::-1]
    return contour


def filter_and_scale_contours(contours, domain_w, domain_h, letter_height,
                              min_area_ratio=0.001):
    """Scale contours to fit domain and filter out tiny ones."""
    # Get bounding box of all contours
    all_pts = np.vstack(contours)
    xmin, ymin = all_pts.min(axis=0)
    xmax, ymax = all_pts.max(axis=0)
    text_w = xmax - xmin
    text_h = ymax - ymin

    if text_h < 1e-10:
        raise ValueError("Text has zero height")

    # Scale to desired letter_height, center in domain
    scale = letter_height / text_h
    cx = domain_w / 2
    cy = domain_h / 2
    text_cx = (xmin + xmax) / 2
    text_cy = (ymin + ymax) / 2

    total_area = text_w * text_h * scale * scale
    scaled = []
    for c in contours:
        sc = (c - [text_cx, text_cy]) * scale + [cx, cy]
        # Compute area
        x, y = sc[:-1, 0], sc[:-1, 1]
        area = abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        if area / total_area > min_area_ratio:
            scaled.append(sc)

    return scaled


def write_control_file(contours, domain_w, domain_h, element_size,
                       output_dir, poly_order=4):
    """Write HOHQMesh control file with icm inner boundaries."""
    ctrl_path = f"{output_dir}/icm.control"
    mesh_name = f"{output_dir}/icm"

    with open(ctrl_path, "w") as f:
        f.write("%\n")
        f.write("% HOHQMesh control file for 'icm' text mesh\n")
        f.write(f"% Domain: {domain_w}m x {domain_h}m\n")
        f.write(f"% Element size: ~{element_size}m\n")
        f.write("%\n")
        f.write("\\begin{CONTROL_INPUT}\n\n")

        # Run parameters
        f.write("   \\begin{RUN_PARAMETERS}\n")
        f.write(f"      mesh file name   = {mesh_name}.inp\n")
        f.write(f"      plot file name   = {mesh_name}.tec\n")
        f.write(f"      stats file name  = {mesh_name}.txt\n")
        f.write("      mesh file format = ABAQUS\n")
        f.write(f"      polynomial order = {poly_order}\n")
        f.write("      plot file format = skeleton\n")
        f.write("   \\end{RUN_PARAMETERS}\n\n")

        # Background grid (explicit box)
        f.write("   \\begin{BACKGROUND_GRID}\n")
        f.write("       x0 = [0.0, 0.0, 0.0]\n")
        dx = element_size
        nx = int(np.ceil(domain_w / dx))
        ny = int(np.ceil(domain_h / dx))
        f.write(f"       dx = [{dx}, {dx}, 0.0]\n")
        f.write(f"       N  = [{nx}, {ny}, 1]\n")
        f.write("   \\end{BACKGROUND_GRID}\n\n")

        # Smoother
        f.write("   \\begin{SPRING_SMOOTHER}\n")
        f.write("      smoothing            = ON\n")
        f.write("      smoothing type       = LinearAndCrossBarSpring\n")
        f.write("      number of iterations = 40\n")
        f.write("   \\end{SPRING_SMOOTHER}\n\n")

        f.write("\\end{CONTROL_INPUT}\n\n")

        # Model with inner boundaries
        f.write("\\begin{MODEL}\n\n")
        f.write("   \\begin{INNER_BOUNDARIES}\n\n")

        for i, contour in enumerate(contours):
            contour = ensure_ccw(contour)
            n = len(contour)
            t_vals = np.linspace(0, 1, n)

            f.write(f"      \\begin{{CHAIN}}\n")
            f.write(f"         name = letter_boundary_{i+1}\n")
            f.write(f"         \\begin{{SPLINE_CURVE}}\n")
            f.write(f"            name   = curve_{i+1}\n")
            f.write(f"            nKnots = {n}\n")
            f.write(f"            \\begin{{SPLINE_DATA}}\n")
            for j in range(n):
                f.write(f"               {t_vals[j]:.15e}  "
                        f"{contour[j, 0]:.15e}  "
                        f"{contour[j, 1]:.15e}  "
                        f"0.0\n")
            f.write(f"            \\end{{SPLINE_DATA}}\n")
            f.write(f"         \\end{{SPLINE_CURVE}}\n")
            f.write(f"      \\end{{CHAIN}}\n\n")

        f.write("   \\end{INNER_BOUNDARIES}\n\n")
        f.write("\\end{MODEL}\n")
        f.write("\\end{FILE}\n")

    print(f"Control file written: {ctrl_path}")
    print(f"  Domain: {domain_w} x {domain_h} m")
    print(f"  Grid: {nx} x {ny} elements (h = {dx} m)")
    print(f"  Inner boundaries: {len(contours)}")
    return ctrl_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate HOHQMesh control file for 'icm' text mesh")
    parser.add_argument("--domain-w", type=float, default=6000.0,
                        help="Domain width [m]")
    parser.add_argument("--domain-h", type=float, default=3000.0,
                        help="Domain height [m]")
    parser.add_argument("--letter-height", type=float, default=1600.0,
                        help="Letter height [m]")
    parser.add_argument("--element-size", type=float, default=80.0,
                        help="Background element size [m]")
    parser.add_argument("--poly-order", type=int, default=4,
                        help="Polynomial order for curved boundaries")
    parser.add_argument("--n-points", type=int, default=150,
                        help="Points per contour for resampling")
    parser.add_argument("--smooth-sigma", type=float, default=3.0,
                        help="Gaussian smoothing sigma (0 to disable)")
    parser.add_argument("--smooth-passes", type=int, default=2,
                        help="Number of smooth-resample passes")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory")
    parser.add_argument("--font", type=str, default="serif",
                        help="Font family")
    args = parser.parse_args()

    print("Extracting letter contours for 'icm'...")
    contours_raw = get_letter_contours("icm", font_size=72, font_family=args.font)
    print(f"  Raw contours: {len(contours_raw)}")

    # Resample each contour
    contours_resampled = []
    for c in contours_raw:
        r = resample_contour(c, n_points=args.n_points)
        if r is not None:
            contours_resampled.append(r)
    print(f"  Resampled contours: {len(contours_resampled)}")

    # Iterative smooth + resample for quality
    if args.smooth_sigma > 0:
        for p in range(args.smooth_passes):
            smoothed = []
            for c in contours_resampled:
                c = smooth_contour(c, sigma=args.smooth_sigma)
                c = resample_contour(c, n_points=args.n_points)
                if c is not None:
                    smoothed.append(c)
            contours_resampled = smoothed
        print(f"  After {args.smooth_passes} smoothing passes "
              f"(sigma={args.smooth_sigma}): {len(contours_resampled)} contours")

    # Scale and filter
    contours_scaled = filter_and_scale_contours(
        contours_resampled, args.domain_w, args.domain_h,
        args.letter_height, min_area_ratio=0.005)
    print(f"  Scaled contours (filtered): {len(contours_scaled)}")

    # Remove the dot of 'i' — find the contour with highest centroid y
    # and smallest area among those with high y
    if len(contours_scaled) > 3:
        centroids_y = [c[:-1, 1].mean() for c in contours_scaled]
        areas = []
        for c in contours_scaled:
            x, y = c[:-1, 0], c[:-1, 1]
            areas.append(abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))

        # The dot is the small contour with highest y centroid
        max_cy = max(centroids_y)
        dot_candidates = [(idx, areas[idx]) for idx in range(len(contours_scaled))
                          if centroids_y[idx] > max_cy - 200]  # within 200m of top
        if dot_candidates:
            dot_idx = min(dot_candidates, key=lambda x: x[1])[0]
            dot_center = contours_scaled[dot_idx][:-1].mean(axis=0)
            print(f"  Removed 'i' dot (contour {dot_idx}), "
                  f"center: ({dot_center[0]:.0f}, {dot_center[1]:.0f})")
            print(f"  -> Source position for simulation: "
                  f"({dot_center[0]:.0f}, {dot_center[1]:.0f})")
            # Save dot position
            np.savetxt(f"{args.output_dir}/source_position.txt",
                       dot_center.reshape(1, -1),
                       header="x y (position of i-dot, use as source location)")
            contours_scaled.pop(dot_idx)

    # Print contour info
    for i, c in enumerate(contours_scaled):
        cx, cy = c[:-1].mean(axis=0)
        x, y = c[:-1, 0], c[:-1, 1]
        area = abs(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        print(f"  Contour {i+1}: center=({cx:.0f}, {cy:.0f}), "
              f"area={area:.0f} m², points={len(c)}")

    # Write control file
    write_control_file(contours_scaled, args.domain_w, args.domain_h,
                       args.element_size, args.output_dir, args.poly_order)


if __name__ == "__main__":
    main()
