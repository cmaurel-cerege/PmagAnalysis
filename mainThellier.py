## This was written under supervision by Claude AI

"""
Thellier-Thellier IZZI experiment plotting tool
=================================================

Reads a data file with columns:
    step   Mx   My   Mz

where `step` encodes temperature + step type as  TEMP.CODE :
    .00  -> Z step   (zero-field demag step, gives NRM remaining)
    .01  -> I step   (in-field step, gives NRM remaining + pTRM gained)
    .02  -> pTRM check (repeat of an earlier in-field step)
    .03  -> pTRM tail check (repeat of a zero-field step, MD check)

Produces up to four figures:
    1. Zijderveld diagram (Z steps only)
    2. Arai plot: normalized NRM remaining vs normalized pTRM gained
       (with pTRM checks overlaid, if present)
    3. Normalized NRM (Z) and pTRM (I) vs temperature
    4. MD (pTRM tail) check plot: normalized tail difference vs temperature
       (only produced if .03 steps are present in the data)

Optionally, you can select a temperature interval [Tmin, Tmax] on the
Arai plot and fit a best-fit line through the Arai points in that range,
reporting the slope (and other paleointensity statistics). This can be
done either by passing --tmin/--tmax on the command line, or by running
the script without them and answering the interactive prompt after the
figures are generated.

Usage:
    python thellier_plots.py <datafile> [--out OUTDIR] [--tmin T] [--tmax T]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
from plotZijderveld import *
import sys

STEP_NAMES = {0: "Z", 1: "I", 2: "pTRM check", 3: "pTRM tail check"}


def load_data(path):
    df = pd.read_csv(path, sep=r"\s+", header=None,
                      names=["step", "Mx", "My", "Mz"], engine="python")

    # temperature = integer part, code = the two-digit decimal part
    # (use rounding to avoid float误差, e.g. 195.01 -> temp 195, code 1)
    df["temp"] = np.floor(df["step"] + 1e-6).astype(int)
    df["code"] = np.round((df["step"] - df["temp"]) * 100).astype(int)
    df["step_type"] = df["code"].map(STEP_NAMES).fillna("unknown")

    # preserve original measurement order (the file order = the true
    # chronological order of measurement, which is not sorted by temp
    # because pTRM checks / tail checks are interleaved)
    df["seq"] = np.arange(len(df))

    # If a given (temp, code) combination appears more than once,
    # keep the LAST measurement (assumed to supersede earlier ones)
    df = df.sort_values("seq").drop_duplicates(subset=["temp", "code"], keep="last")

    df["M"] = np.sqrt(df["Mx"] ** 2 + df["My"] ** 2 + df["Mz"] ** 2)

    return df.sort_values(["temp", "code"]).reset_index(drop=True)


def build_series(df):
    """Split into the four step types, each sorted by temperature."""
    Z = df[df.code == 0].sort_values("temp").reset_index(drop=True)
    I = df[df.code == 1].sort_values("temp").reset_index(drop=True)
    PC = df[df.code == 2].sort_values("temp").reset_index(drop=True)
    TAIL = df[df.code == 3].sort_values("temp").reset_index(drop=True)
    return Z, I, PC, TAIL


def compute_arai(Z, I, NRM0):
    """
    Build the Arai-plot table: for every temperature that has BOTH a
    Z step and an I step, compute pTRM gained as the vector difference
    I - Z, and pair it with the NRM remaining (Z step).
    """
    merged = pd.merge(Z, I, on="temp", suffixes=("_Z", "_I"))
    merged["pTRMx"] = merged["Mx_I"] - merged["Mx_Z"]
    merged["pTRMy"] = merged["My_I"] - merged["My_Z"]
    merged["pTRMz"] = merged["Mz_I"] - merged["Mz_Z"]
    merged["pTRM"] = np.sqrt(merged.pTRMx**2 + merged.pTRMy**2 + merged.pTRMz**2)
    merged["NRM_norm"] = merged["M_Z"] / NRM0
    merged["pTRM_norm"] = merged["pTRM"] / NRM0
    return merged.sort_values("temp").reset_index(drop=True)


def compute_pc_norm(PC, Z, arai, NRM0):
    """
    Normalized pTRM-check points, for overplotting on the Arai plot.

    IMPORTANT: a pTRM check at temperature Ti is measured AFTER the
    specimen has already been thermally demagnetized up to some higher
    temperature Tn > Ti (that's the point of the check - to see if
    heating to Tn altered the pTRM capacity at the lower temperature
    Ti). By the time the check happens, everything with unblocking
    temperature between Ti and Tn has already been erased by the
    intervening zero-field steps. So the correct zero-field baseline to
    subtract is the MOST RECENT Z step actually measured before the
    check (at Tn) - NOT the original Z step back at Ti. Using the
    original (older) Ti baseline overstates the check's apparent pTRM,
    since it still contains magnetization that no longer exists in the
    specimen at the time of the check.

    We use the true chronological order of measurements (the file's
    row order, preserved as `seq`) to find, for each check, the most
    recent Z step performed before it.
    """
    if PC.empty:
        return None

    Z_by_seq = Z.sort_values("seq").reset_index(drop=True)
    records = []
    for _, row in PC.iterrows():
        prior = Z_by_seq[Z_by_seq.seq < row.seq]
        if prior.empty:
            continue  # no zero-field baseline was measured yet - can't evaluate
        baseline = prior.iloc[-1]  # most recent (highest-temp) Z step so far
        dx = row.Mx - baseline.Mx
        dy = row.My - baseline.My
        dz = row.Mz - baseline.Mz
        records.append({
            "temp": row.temp,               # temperature being checked (Ti)
            "baseline_temp": baseline.temp,  # temperature of the Z baseline used (Tn)
            "pTRM_check_norm": np.sqrt(dx**2 + dy**2 + dz**2) / NRM0,
        })

    if not records:
        return None

    out = pd.DataFrame(records)
    out = out.merge(arai[["temp", "NRM_norm"]], on="temp", how="left")
    # also bring in the Arai-plot coordinates of the baseline Z step itself,
    # so we can draw a connector line from that point to the check triangle
    baseline_coords = arai[["temp", "pTRM_norm", "NRM_norm"]].rename(
        columns={"temp": "baseline_temp", "pTRM_norm": "baseline_pTRM_norm",
                 "NRM_norm": "baseline_NRM_norm"})
    out = out.merge(baseline_coords, on="baseline_temp", how="left")
    return out


def compute_md_check(TAIL, Z, NRM0):
    """
    Normalized MD (pTRM tail) check values, as a function of temperature.

    A pTRM tail check at temperature Tc repeats the ZERO-FIELD step at Tc,
    after the specimen has already been through the in-field step at Tc
    (and possibly higher-temperature steps). Comparing it to the original
    Z-step measurement at Tc reveals any magnetization "tail" left behind
    by multidomain-like (MD) behaviour or alteration.

    Returns a DataFrame with columns: temp, tail_diff (vector-difference
    magnitude between the tail check and the original Z step) and
    tail_diff_norm (normalized by NRM0), sorted by temperature.
    """
    if TAIL.empty:
        return pd.DataFrame(columns=["temp", "tail_diff", "tail_diff_norm"])

    out = pd.merge(TAIL, Z[["temp", "Mx", "My", "Mz"]], on="temp",
                    suffixes=("_tail", "_Zorig"))
    if out.empty:
        return pd.DataFrame(columns=["temp", "tail_diff", "tail_diff_norm"])

    dx = out["Mx_tail"] - out["Mx_Zorig"]
    dy = out["My_tail"] - out["My_Zorig"]
    dz = out["Mz_tail"] - out["Mz_Zorig"]
    out["tail_diff"] = np.sqrt(dx**2 + dy**2 + dz**2)
    out["tail_diff_norm"] = out["tail_diff"] / NRM0

    return out[["temp", "tail_diff", "tail_diff_norm"]].sort_values("temp").reset_index(drop=True)


def compute_linear_fit(arai, tmin, tmax, field):
    """
    Fit a best-fit line through the Arai points with tmin <= temp <= tmax,
    using the standard paleointensity "line fit" (Coe, 1978): this is a
    major-axis / total-least-squares fit (treating NRM and pTRM as having
    comparable uncertainty), NOT an ordinary least-squares regression.

        slope = sign(Sxy) * sqrt(Syy / Sxx)
        intercept = ybar - slope * xbar

    where x = normalized pTRM gained, y = normalized NRM remaining, and
    Sxx, Syy, Sxy are the sums of squares/cross-products about the means.

    The standard error of the slope follows Coe, Gromme & Mankinen (1978)
    (also used in the Standard Paleointensity Definitions, Paterson et
    al. 2014, as the basis of the "beta" quality statistic):

        sigma_b = sqrt( 2 * sum(e_i^2) / ((n - 2) * Sxx) )

    where e_i = y_i - (slope*x_i + intercept) are the vertical residuals
    from the best-fit line. A 95% confidence interval on the slope is
    then slope +/- t(0.975, n-2) * sigma_b, using the Student's
    t-distribution (needs n >= 3 points; with only 2 points the slope is
    exact and has no defined uncertainty).

    Returns a dict with the fit results, or None if fewer than 2 points
    fall in the requested range.
    """
    sel = arai[(arai.temp >= tmin) & (arai.temp <= tmax)].sort_values("temp").reset_index(drop=True)
    if len(sel) < 2:
        return None

    x = sel.pTRM_norm.values
    y = sel.NRM_norm.values
    n = len(sel)
    xbar, ybar = x.mean(), y.mean()
    Sxx = np.sum((x - xbar) ** 2)
    Syy = np.sum((y - ybar) ** 2)
    Sxy = np.sum((x - xbar) * (y - ybar))

    if Sxx == 0:
        return None

    slope = np.sign(Sxy) * np.sqrt(Syy / Sxx) if Syy > 0 else 0.0
    intercept = ybar - slope * xbar
    r = Sxy / np.sqrt(Sxx * Syy) if Sxx > 0 and Syy > 0 else np.nan

    se_slope = np.nan
    ci95 = (np.nan, np.nan)
    t_crit = np.nan
    if n >= 3:
        y_pred = slope * x + intercept
        sse = np.sum((y - y_pred) ** 2)
        se_slope = np.sqrt(2 * sse / ((n - 2) * Sxx))
        t_crit = stats.t.ppf(0.975, df=n - 2)
        margin = t_crit * se_slope
        ci95 = (slope - margin, slope + margin)

    return {
        "tmin": tmin,
        "tmax": tmax,
        "n_points": n,
        "temps": sel.temp.tolist(),
        "x_range": (x.min(), x.max()),
        "slope": slope,
        "intercept": intercept,
        "r": r,
        "paleointensity_ratio": abs(slope)*field,  # |slope| = B_ancient / B_lab
        "se_slope": se_slope,
        "ci95": ci95,
        "t_crit": t_crit,
    }



# def plot_zijderveld(Z):
#     fig, ax = plt.subplots(figsize=(6.5, 6.5))
#
#     # Horizontal projection (Mx vs My) and Vertical projection (Mx vs -Mz)
#     ax.plot(Z.Mx, Z.My, "o-", color="black", mfc="black",
#             label="Horizontal (X-Y)", markersize=5)
#     ax.plot(Z.Mx, -Z.Mz, "s-", color="black", mfc="white",
#             label="Vertical (X, -Z)", markersize=5)
#
#     for _, row in Z.iterrows():
#         ax.annotate(f"{row.temp:g}", (row.Mx, row.My), fontsize=7,
#                     textcoords="offset points", xytext=(4, 4), color="dimgray")
#
#     ax.axhline(0, color="gray", lw=0.5)
#     ax.axvline(0, color="gray", lw=0.5)
#     ax.set_xlabel("Mx (Am$^2$)")
#     ax.set_ylabel("My , -Mz  (Am$^2$)")
#     ax.set_title("Zijderveld diagram (Z steps)")
#     ax.legend(loc="best", fontsize=9)
#     ax.set_aspect("equal", adjustable="datalim")
#     fig.tight_layout()

def plot_arai(arai, pc_norm, fit=None):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.plot(arai.pTRM_norm, arai.NRM_norm, "o-", color="black", mfc="royalblue",markersize=7,zorder=3)

    for _, row in arai.iterrows():
        ax.annotate(f"{row.temp:g}", (row.pTRM_norm, row.NRM_norm), fontsize=7,
                    textcoords="offset points", xytext=(5, 5), color="dimgray")

    if pc_norm is not None and not pc_norm.empty:
        # connector line: baseline Z step's Arai point -> corner -> check triangle
        # (horizontal then vertical "dog-leg", as is conventional)
        first_line = True
        for _, row in pc_norm.iterrows():
            if pd.isna(row.baseline_pTRM_norm) or pd.isna(row.baseline_NRM_norm):
                continue  # baseline has no matching Arai point (no I step yet) - skip connector
            bx, by = row.baseline_pTRM_norm, row.baseline_NRM_norm
            cx, cy = row.pTRM_check_norm, row.NRM_norm
            #ax.plot([bx, cx], [by, by], "--", color="darkred", lw=1)
            #ax.plot([cx, cx], [by, cy], "--", color="darkred", lw=1)
            first_line = False

        ax.plot(pc_norm.pTRM_check_norm, pc_norm.NRM_norm, "^", color="darkred",
                markersize=9, mfc="none", mew=1.5)
        for _, row in pc_norm.iterrows():
            ax.annotate(f"{row.temp:g}",
                        (row.pTRM_check_norm, row.NRM_norm),
                        fontsize=7, textcoords="offset points", xytext=(5, -8),
                        color="darkred")

    if fit is not None:
        sel_mask = arai.temp.isin(fit["temps"])
        #ax.plot(arai.pTRM_norm[sel_mask], arai.NRM_norm[sel_mask], "o", color="seagreen", mfc="seagreen", markersize=9, zorder=5)

        x0, x1 = fit["x_range"]
        y0 = fit["slope"] * x0 + fit["intercept"]
        y1 = fit["slope"] * x1 + fit["intercept"]
        ax.plot([x0, x1], [y0, y1], "--", color="seagreen", lw=2.5,label=f"Fit {fit['tmin']:g}-{fit['tmax']:g}\N{DEGREE SIGN}C")

        if np.isfinite(fit["se_slope"]):
            ci_lo, ci_hi = fit["ci95"]
            fit_text = (f"slope = {fit['slope']:.3f}\n"
                        f"2 s.e.= {2*fit['se_slope']:.4f}"
                        #f"95% CI = [{ci_lo:.3f}, {ci_hi:.3f}]\n"
                        f"n = {fit['n_points']},  r = {fit['r']:.3f}")
        else:
            fit_text = f"slope = {fit['slope']:.3f}\nn = {fit['n_points']},  r = {fit['r']:.3f}"
        ax.text(0.03, 0.03, fit_text, transform=ax.transAxes, fontsize=9,
                color="darkgray", va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="darkgray"))

    ax.set_xlabel("Normalized pTRM gained")
    ax.set_ylabel("Normalized NRM remaining")
    ax.set_title("Arai plot")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()


def build_temp_series(Z, arai):
    """
    Table used for the NRM/pTRM vs temperature plot: the normal Arai-plot
    points, PLUS the starting point (the very first Z step, typically
    room temperature / lowest T), where by definition NRM_norm = 1 and
    no pTRM has been imparted yet, so pTRM_norm = 0.
    """
    start_temp = Z.iloc[0].temp
    start = pd.DataFrame({"temp": [start_temp], "NRM_norm": [1.0], "pTRM_norm": [0.0]})
    combined = pd.concat([start, arai[["temp", "NRM_norm", "pTRM_norm"]]], ignore_index=True)
    return combined.sort_values("temp").reset_index(drop=True)


def plot_vs_temperature(temp_series):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(temp_series.temp, temp_series.NRM_norm, "o-", color="royalblue", label="Normalized NRM (Z steps)")
    ax.plot(temp_series.temp, temp_series.pTRM_norm, "s-", color="firebrick", label="Normalized pTRM (I steps)")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Normalized magnetization")
    ax.set_title("Normalized NRM and pTRM vs temperature")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()


def plot_md_check(md):
    """
    MD (pTRM tail) check plot: normalized tail difference vs temperature.
    If no tail-check data is present, still produce a plot with an
    explanatory note rather than skipping it, so the figure set stays
    consistent across datasets.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    if md.empty:
        ax.text(0.5, 0.5, "No pTRM tail check (.03) data in this file",
                ha="center", va="center", fontsize=11, color="dimgray",
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.plot(md.temp, md.tail_diff_norm, "D-", color="seagreen", markersize=7)
        for _, row in md.iterrows():
            ax.annotate(f"{row.temp:g}", (row.temp, row.tail_diff_norm), fontsize=7,
                        textcoords="offset points", xytext=(5, 5), color="dimgray")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Normalized tail difference  |M$_{tail}$ - M$_Z$| / NRM$_0$")
        ax.set_ylim(bottom=0)

    ax.set_title("MD (pTRM tail) check")
    fig.tight_layout()


def main():

    file = sys.argv[1]
    path = ''
    fp = open(file, 'r')
    for k in np.arange(len(file.split('/')) - 1):
        path += str(file.split('/')[k]) + '/'
    if not os.path.exists(path + 'Plots'):
        os.makedirs(path + 'Plots')
    sample = file.split('/')[-1].split('.')[0]

    save = input('Save the figures? (y/N)')

    field = input('Applied field intensity? (default = 10 uT)  ')
    if field == '':
        field = 10
    else:
        field = float(eval(field))

    parser = argparse.ArgumentParser(description="Plot Thellier-Thellier IZZI data")
    parser.add_argument("datafile")
    parser.add_argument("--out", default=".", help="output directory")
    parser.add_argument("--tmin", type=float, default=None,
                         help="lower temperature bound (°C) for the Arai linear fit")
    parser.add_argument("--tmax", type=float, default=None,
                         help="upper temperature bound (°C) for the Arai linear fit")
    parser.add_argument("--no-prompt", action="store_true",
                         help="skip the interactive fit-bounds prompt if --tmin/--tmax not given")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.datafile)
    Z, I, PC, TAIL = build_series(df)

    if Z.empty:
        raise SystemExit("No Z steps (.00) found in the data - cannot proceed.")

    NRM0 = Z.iloc[0].M  # magnitude of the first (lowest T) Z step = total NRM
    NRMx, NRMy, NRMz = np.array(Z.iloc[:].Mx), np.array(Z.iloc[:].My), np.array(Z.iloc[:].Mz)
    Thstep = np.array(Z.iloc[:].step)

    arai = compute_arai(Z, I, NRM0)
    pc_norm = compute_pc_norm(PC, Z, arai, NRM0) if not PC.empty else None
    md = compute_md_check(TAIL, Z, NRM0)
    temp_series = build_temp_series(Z, arai)

    base = os.path.splitext(os.path.basename(args.datafile))[0]

    #Plot_Zijderveld(NRMx, NRMy, NRMz, Thstep, unit='A m2', title='NRM@TH', color='k')
    # if save == 'y':
    #     plt.savefig(path+'Plots/'+sample + '-TH-Zijd.pdf', format='pdf', dpi=400, bbox_inches="tight")
    Plot_Zijderveld(NRMx, NRMy, NRMz, Thstep, unit='A m2', title='NRM@TH', color='k', gui='guiZ',annot='X')

    plot_arai(arai, pc_norm)
    if save == 'y':
        plt.savefig(path+'Plots/'+sample + '-TH-Arai.pdf', format='pdf', dpi=400, bbox_inches="tight")
    plot_vs_temperature(temp_series)
    if save == 'y':
        plt.savefig(path+'Plots/'+sample + '-TH-Demag.pdf', format='pdf', dpi=400, bbox_inches="tight")
    plot_md_check(md)
    if save == 'y':
        plt.savefig(path+'Plots/'+sample + '-TH-MDcheck.pdf', format='pdf', dpi=400, bbox_inches="tight")

    print(arai[["temp", "NRM_norm", "pTRM_norm"]].to_string(index=False))
    if pc_norm is not None and not pc_norm.empty:
        print("\npTRM checks (baseline = most recent Z step measured before the check):")
        print(pc_norm[["temp", "baseline_temp", "pTRM_check_norm"]].to_string(index=False))
    if not md.empty:
        print("\nMD (pTRM tail) checks:")
        print(md.to_string(index=False))
    else:
        print("\nNo pTRM tail check (.03) steps found in this file.")

    # --- Optional linear fit over a chosen temperature interval ---
    tmin, tmax = args.tmin, args.tmax
    if tmin is None and tmax is None and not args.no_prompt:
        print(f"\nAvailable Arai temperature steps: {arai.temp.tolist()}")
        try:
            tmin_in = input("Enter lower temperature bound for a linear fit "
                             "(or press Enter to skip): ").strip()
            if tmin_in:
                tmax_in = input("Enter upper temperature bound (default: last step): ").strip()
                if tmax_in == '':
                    tmax_in = Thstep[-1]
                tmin, tmax = float(tmin_in), float(tmax_in)
        except (EOFError, ValueError):
            tmin, tmax = None, None

    if tmin is not None and tmax is not None:
        lo, hi = min(tmin, tmax), max(tmin, tmax)
        fit = compute_linear_fit(arai, lo, hi,field)
        if fit is None:
            print(f"\nNot enough Arai points between {lo:g} and {hi:g} °C to fit a line "
                  "(need at least 2).")
        else:
            print(f"\nLinear fit between {lo:g}-{hi:g}\u00b0C "
                  f"(n={fit['n_points']} points: {fit['temps']}):")
            print(f"  slope               = {fit['slope']:.4f}")
            print(f"  intercept           = {fit['intercept']:.4f}")
            print(f"  correlation r       = {fit['r']:.4f}")
            print(f"  pint                = {fit['paleointensity_ratio']:.4f}")
            if np.isfinite(fit["se_slope"]):
                ci_lo, ci_hi = fit["ci95"]
                print(f"  2 s.e.          = {2*fit['se_slope']*field:.4f}")
                #print(f"  95% CI pint     = [{ci_lo*field:.4f}, {ci_hi*field:.4f}]  "
                      #f"(t={fit['t_crit']:.3f}, dof={fit['n_points']-2})")
            else:
                print("  standard error / CI = not defined (need >= 3 points)")
            plot_arai(arai, pc_norm, fit=fit)

            if save == 'y':
                plt.savefig(path + 'Plots/' + sample + '-TH-Arai-fit.pdf', format='pdf', dpi=400, bbox_inches="tight")
            plot_vs_temperature(temp_series)
    plt.show()


if __name__ == "__main__":
    main()
