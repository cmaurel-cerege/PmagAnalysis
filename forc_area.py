#!/usr/bin/env python3
"""
forc_area.py

Analyze a MicroMag 2900/3900 FORC data file (First-Order Reversal Curves).

Computes:
  1) The area between a chosen reversal curve (by default, the one whose
     starting/reversal field Hb is closest to a target field, default 0 T)
     and the upper envelope of all FORC curves (max M(H) across all curves).
  2) The total area enclosed within the FORC envelope: the area between the
     upper envelope (max M at each H) and the lower envelope (min M at each
     H), integrated over the field range where curves overlap.

Both areas are computed by interpolating each measured reversal curve onto a
common field grid, then integrating with the trapezoidal rule. Because M is
in A*m^2 and H is in T, the resulting areas have units of A*m^2*T = Joules.

Usage:
    python forc_area.py path/to/forcfile.txt
    python forc_area.py path/to/forcfile.txt --target-h 0.0 --npoints 2000
    python forc_area.py path/to/forcfile.txt --plot out.png

Can also be imported and used programmatically:
    from forc_area import parse_forc_file, analyze_forc
    forcs = parse_forc_file("file.txt")
    results = analyze_forc(forcs)
"""

import argparse
import sys
import numpy as np


def parse_forc_file(path):
    """
    Parse a MicroMag 2900/3900 FORC data file.

    The file has a text header, then a data section where reversal curves
    are separated by blank lines. Single-point groups are calibration /
    saturation check points inserted by the VSM software between curves
    and are excluded from the returned list.

    Returns
    -------
    forcs : list of numpy.ndarray, shape (n_i, 2)
        Each array holds the (H, M) pairs of one reversal curve, sorted by
        ascending H.
    """
    with open(path, "r") as f:
        lines = f.readlines()

    # Find start of numeric data: the line right after the "(T) (A*m^2)"
    # units header.
    start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("(") and "A" in s:
            start = i + 1
            break
    if start is None:
        raise ValueError(
            "Could not find start of data section (expected a units header "
            "line like '(T)\\t\\t(A\\u00b7m\\u00b2)')."
        )

    # Find end of numeric data (the file-end marker line).
    end = len(lines)
    for i, line in enumerate(lines[start:], start=start):
        if "ends" in line.lower():
            end = i
            break

    data_lines = lines[start:end]

    curves = []
    current = []
    for line in data_lines:
        s = line.strip()
        if s == "":
            if current:
                curves.append(current)
                current = []
            continue
        parts = s.split(",")
        if len(parts) < 2:
            continue
        h, m = float(parts[0]), float(parts[1])
        current.append((h, m))
    if current:
        curves.append(current)

    # Keep only actual reversal curves (>1 point); drop single-point
    # calibration/saturation checks.
    forcs = []
    for c in curves:
        if len(c) > 1:
            arr = np.array(c, dtype=float)
            order = np.argsort(arr[:, 0])
            forcs.append(arr[order])

    if not forcs:
        raise ValueError("No multi-point reversal curves were found in the file.")

    return forcs


def find_curve_nearest_start(forcs, target_h=0.0):
    """Return the index of the curve whose starting field is closest to target_h."""
    starts = np.array([c[0, 0] for c in forcs])
    return int(np.argmin(np.abs(starts - target_h)))


def build_envelopes(forcs, grid):
    """
    Build the upper and lower envelopes of a set of FORC curves on a common
    field grid.

    For each grid point, the upper (lower) envelope is the maximum
    (minimum) M value among all curves whose own measured field range
    covers that grid point, using linear interpolation within each curve.

    Points on the grid not covered by any curve are set to NaN.
    """
    upper = np.full_like(grid, -np.inf)
    lower = np.full_like(grid, np.inf)
    covered = np.zeros_like(grid, dtype=bool)

    for c in forcs:
        h, m = c[:, 0], c[:, 1]
        hlo, hhi = h[0], h[-1]
        mask = (grid >= hlo) & (grid <= hhi)
        if not mask.any():
            continue
        interp_m = np.interp(grid[mask], h, m)
        upper[mask] = np.maximum(upper[mask], interp_m)
        lower[mask] = np.minimum(lower[mask], interp_m)
        covered[mask] = True

    upper[~covered] = np.nan
    lower[~covered] = np.nan
    return upper, lower, covered


def analyze_forc(forcs, target_h=0.0, npoints=2000, curve_index=None):
    """
    Run the full FORC area analysis.

    Parameters
    ----------
    forcs : list of (H, M) arrays, as returned by parse_forc_file.
    target_h : float
        Field (T) used to pick the reference reversal curve, if
        curve_index is not given directly. The curve whose starting field
        is closest to target_h is used.
    npoints : int
        Number of points in the common interpolation grid.
    curve_index : int or None
        If given, use this curve (index into `forcs`) instead of searching
        by target_h.

    Returns
    -------
    dict with keys:
        curve_index, curve, grid, upper, lower, covered,
        area_curve_to_upper, area_total_envelope
    """
    if curve_index is None:
        curve_index = find_curve_nearest_start(forcs, target_h)
    target = forcs[curve_index]
    th, tm = target[:, 0], target[:, 1]

    # Grid over the full range spanned by all curves, so the "total
    # envelope area" reflects the whole FORC fan, not just one curve's range.
    all_h = np.concatenate([c[:, 0] for c in forcs])
    grid = np.linspace(all_h.min(), all_h.max(), npoints)

    upper, lower, covered = build_envelopes(forcs, grid)

    # Area between the target curve and the upper envelope, restricted to
    # the target curve's own field range.
    in_range = (grid >= th[0]) & (grid <= th[-1]) & covered
    target_on_grid = np.interp(grid[in_range], th, tm)
    area_curve_to_upper = np.trapezoid(
        upper[in_range] - target_on_grid, grid[in_range]
    )

    # Total area enclosed within the FORC envelope (upper - lower),
    # over the full covered field range.
    area_total_envelope = np.trapezoid(
        np.where(covered, upper - lower, 0.0), grid
    )

    return {
        "curve_index": curve_index,
        "curve": target,
        "grid": grid,
        "upper": upper,
        "lower": lower,
        "covered": covered,
        "area_curve_to_upper": area_curve_to_upper,
        "area_total_envelope": area_total_envelope,
    }


def plot_results(results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = results["grid"]
    upper = results["upper"]
    lower = results["lower"]
    th, tm = results["curve"][:, 0], results["curve"][:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=150)

    ax = axes[0]
    ax.plot(grid, upper, color="black", linewidth=1.2, label="Upper envelope")
    ax.plot(grid, lower, color="gray", linewidth=1.2, label="Lower envelope")
    ax.fill_between(grid, lower, upper, color="steelblue", alpha=0.3,
                     label=f"Total envelope area = {results['area_total_envelope']:.3e} J")
    ax.set_xlabel("Field, H (T)")
    ax.set_ylabel("Moment, M (A\u00b7m\u00b2)")
    ax.set_title("Total area within the FORC envelope")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(grid, upper, color="black", linewidth=1.2, label="Upper envelope")
    ax2.plot(th, tm, color="crimson", linewidth=1.5,
              label=f"Curve #{results['curve_index']} (start H={th[0]:.4f} T)")
    ax2.fill_between(th, tm, np.interp(th, grid, upper), color="orange", alpha=0.4,
                      label=f"Area to envelope = {results['area_curve_to_upper']:.3e} J")
    ax2.set_xlabel("Field, H (T)")
    ax2.set_ylabel("Moment, M (A\u00b7m\u00b2)")
    ax2.set_title("Area between chosen curve and upper envelope")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def get_energy_parameters(path):
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("forc_file", help="Path to the MicroMag FORC .txt data file")
    parser.add_argument("--target-h", type=float, default=0.0,
                         help="Field (T) used to select the reference reversal curve "
                              "(closest starting field is used). Default: 0.0")
    parser.add_argument("--curve-index", type=int, default=None,
                         help="Use this specific curve index instead of searching by --target-h")
    parser.add_argument("--npoints", type=int, default=2000,
                         help="Number of points in the interpolation grid. Default: 2000")
    parser.add_argument("--plot", type=str, default=None,
                         help="If given, save a diagnostic plot to this path (e.g. out.png)")
    args = parser.parse_args()

    forcs = parse_forc_file(args.forc_file)
    results = analyze_forc(
        forcs,
        target_h=args.target_h,
        npoints=args.npoints,
        curve_index=args.curve_index,
    )

    th = results["curve"][:, 0]
    print(f"Parsed {len(forcs)} reversal curves from: {args.forc_file}")
    print(f"Reference curve: index {results['curve_index']}, "
          f"start H = {th[0]:.6f} T, end H = {th[-1]:.6f} T")
    print()
    print(f"1) Area between curve #{results['curve_index']} and the upper envelope: "
          f"{results['area_curve_to_upper']:.6e} A*m^2*T (J)")
    print(f"2) Total area enclosed within the FORC envelope (upper - lower): "
          f"{results['area_total_envelope']:.6e} A*m^2*T (J)")

    Et_delta = 2*results['area_curve_to_upper']
    Ehyst = results['area_total_envelope']

    if args.plot:
        plot_results(results, args.plot)
        print(f"\nSaved diagnostic plot to: {args.plot}")

    return Et_delta, Ehyst

# if __name__ == "__main__":
#     main()
