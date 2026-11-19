"""Figures for the manuscript.  Run after the simulations listed in jobs.txt have finished."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
from common import *

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "xtick.direction": "in",
                     "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
CJ, CS, CI = "#c0392b", "#2c3e50", "#2980b9"      # contact, insulated, independent-electron
DS = [2.3, 3.4, 4.6, 5.7, 6.8, 8.5, 11.4]

def sheet(n, d):
    return areal(n, d) / 1e12

def fields(d):
    """peak forward (clipped) and reverse fields and the quasistatic component (central two cycles), V/nm."""
    F = d["Fin"] * VNM; T = 2 * np.pi / float(d["w"]); t = d["t"] / T; n = d["p"]["ncyc"]
    m = (t > n / 2 - 1) & (t < n / 2 + 1)
    return -F.min(), F.max(), F[m].mean()

def bulk_charge(d, m=2):
    """induced electron number beyond the first m cells at each snapshot (per chain and spin)."""
    z = d["z"]; dx = z[1] - z[0]; a = d["p"]["a"]
    z0 = d["p"]["dsp"] * NM if d["p"]["variant"] == "spacer" else 0.0
    w = 0.5 * (1 + np.tanh((z - z0 - m * a) / (0.5 * a)))
    return d["snaps_t"], (d["snaps_dn"] * w).sum(1) * dx

# ------------------------------------------------------------------ Fig. 1
def fig1():
    fig = plt.figure(figsize=(7.0, 2.15))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.25, 1.1], height_ratios=[1, 1], wspace=0.36, hspace=0.12,
                          left=0.01, right=0.99, top=0.97, bottom=0.17)
    # (a) schematic
    ax = fig.add_subplot(gs[:, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(Rectangle((0.3, 6.0), 9.4, 0.9, fc="#d6dbdf", ec="k", lw=0.5))            # crystal top layer
    ax.add_patch(Rectangle((0.3, 4.9), 9.4, 1.1, fc="#ebedef", ec="k", lw=0.5))
    ax.add_patch(Rectangle((3.2, 6.9), 3.6, 1.0, fc="#f1c40f", ec="k", lw=0.5))            # antenna
    ax.text(5.0, 7.4, "Au", ha="center", va="center", fontsize=7)
    ax.text(0.5, 5.2, "crystal", fontsize=7)
    th = np.linspace(0, 1, 30)
    for sgn, x0 in ((-1, 3.2), (1, 6.8)):
        for r_ in (0.55, 1.0, 1.5):
            ax.plot(x0 + sgn * r_ * (1 - np.cos(th * np.pi / 2)) * 1.2, 6.9 - r_ * np.sin(th * np.pi / 2), color=CJ, lw=0.6)
    ax.add_patch(FancyArrowPatch((6.8, 6.9), (8.55, 5.45), arrowstyle="->", mutation_scale=7, color="k", lw=0.8, connectionstyle="arc3,rad=0.28"))
    ax.text(8.65, 5.15, "$z$", fontsize=8)
    ax.text(5.0, 8.6, r"$E_{\rm L}(t)$", ha="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((3.9, 8.25), (6.1, 8.25), arrowstyle="<->", mutation_scale=6, color="k", lw=0.7))
    # band diagram
    y0 = 0.4
    ax.add_patch(Rectangle((0.6, y0), 2.6, 2.0, fc="#f9e79f", ec="none"))
    ax.plot([0.6, 3.2], [y0 + 2.0, y0 + 2.0], "k", lw=0.8)
    ax.text(0.7, y0 + 2.15, r"$E_{\rm F}$", fontsize=7)
    zz = np.linspace(0, 1, 60)
    bend = 1.5 * (1 - np.exp(-zz / 0.35)) * 0.6
    xs = 3.2 + 6.2 * zz
    ax.plot(xs, y0 + 2.55 - bend, color=CS, lw=0.9); ax.plot(xs, y0 + 0.75 - bend, color=CS, lw=0.9)
    ax.plot([3.2, 3.2], [y0, y0 + 3.3], "k", lw=0.5)
    ax.text(9.5, y0 + 2.55 - bend[-1] + 0.12, "CB", fontsize=6.5, ha="right")
    ax.text(9.5, y0 + 0.75 - bend[-1] - 0.42, "VB", fontsize=6.5, ha="right")
    ax.annotate("", xy=(3.28, y0 + 2.55), xytext=(3.28, y0 + 2.0), arrowprops=dict(arrowstyle="<->", lw=0.5, shrinkA=0, shrinkB=0))
    ax.text(3.4, y0 + 2.75, r"$\Phi_{\rm B}$", fontsize=7)
    ax.add_patch(FancyArrowPatch((2.6, y0 + 1.9), (4.6, y0 + 1.9), arrowstyle="->", mutation_scale=7, color=CJ, lw=1.0))
    ax.text(4.75, y0 + 1.72, r"$e^-$", color=CJ, fontsize=7)
    ax.text(1.9, y0 + 0.9, "metal", ha="center", fontsize=7)
    ax.text(0.2, 9.5, "(a)", fontsize=8)
    # (b) internal field, (c) excess transferred charge
    j, s = load("A_j_D5.7"), load("A_s_D5.7")
    T = 2 * np.pi / float(j["w"])
    axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, 1], sharex=axb)
    axb.plot(s["t"] / T, s["Fin"] * VNM, color=CS, ls="--", label="insulated")
    axb.plot(j["t"] / T, j["Fin"] * VNM, color=CJ, label="contact")
    axb.axhline(0, color="k", lw=0.3)
    axb.set_ylabel(r"$F$ (V/nm)"); axb.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.6, columnspacing=1.0)
    axb.set_ylim(-1.25, 2.1); plt.setp(axb.get_xticklabels(), visible=False)
    tj, qj = bulk_charge(j); ts_, qs_ = bulk_charge(s)
    dq = sheet(qj, j) - np.interp(tj, ts_, sheet(qs_, s))
    axc.plot(tj / T, dq, color=CJ); axc.axhline(0, color="k", lw=0.3)
    axc.set_ylabel(r"$\Delta\sigma$ ($10^{12}$ cm$^{-2}$)"); axc.set_xlabel(r"time (optical cycles)"); axc.set_xlim(0, 5)
    axb.text(0.985, 0.86, "(b)", transform=axb.transAxes, ha="right"); axc.text(0.985, 0.82, "(c)", transform=axc.transAxes, ha="right")
    # (d) induced electron density of the contact in excess of the insulated junction
    axd = fig.add_subplot(gs[:, 2])
    z = j["z"]; a = float(j["p"]["a"]); dx = z[1] - z[0]; dsp = s["p"]["dsp"] * NM
    kz = np.arange(-4 * a, 4 * a + dx, dx); ker = np.exp(-0.5 * (kz / (0.6 * a)) ** 2); ker /= ker.sum()
    stj, sts = j["snaps_t"] / T, s["snaps_t"] / T
    for tt, col, lab in ((2.0, CJ, r"$t=2T$"), (2.5, CI, r"$t=2.5T$")):
        dj = j["snaps_dn"][np.argmin(abs(stj - tt))]
        ds_ = np.interp(z + dsp, s["z"], s["snaps_dn"][np.argmin(abs(sts - tt))])     # align the crystals
        dn = np.convolve(dj - ds_, ker, mode="same") * 2 / float(j["Ac"]) * NM ** 3   # electrons per nm^3
        axd.plot(z / NM, dn * 1e3, color=col, label=lab)
    axd.axvline(0, color="k", lw=0.4); axd.axhline(0, color="k", lw=0.3)
    axd.set_xlim(-1.0, 10); axd.set_ylim(-14, 34); axd.set_xlabel(r"$z$ (nm)")
    axd.set_ylabel(r"$\Delta(\delta n)$ ($10^{-3}$ nm$^{-3}$)")
    axd.legend(frameon=False, loc="upper right"); axd.text(0.9, 0.06, "(d)", transform=axd.transAxes)
    fig.savefig(os.path.join(FIG, "figS1.pdf")); plt.close(fig)

# ------------------------------------------------------------------ Fig. 2
def fig2():
    fig, axs = plt.subplots(2, 1, figsize=(3.375, 2.65), sharex=True, gridspec_kw=dict(hspace=0.06, left=0.15, right=0.98, top=0.98, bottom=0.12))
    orders = np.arange(2, 13)
    for ax, D, lab in zip(axs, (5.7, 8.5), ("(a)", "(b)")):
        j, s = load(f"A_j_D{D}"), load(f"A_s_D{D}")
        hj, hs = harmonics(j, range(1, 13)), harmonics(s, range(1, 13))
        Eg = (float(j["Ec"]) - float(j["Ev"])) / float(j["w"])
        ax.axvspan(Eg, 13, color="#ecf0f1", lw=0)
        for n in orders[::2]:
            ax.axvline(n, color="#bdc3c7", lw=0.5, ls=":")
        ax.semilogy(orders, hs[1:] / hs[0], "s--", color=CS, ms=3.5, mfc="w", label="insulated")
        ax.semilogy(orders, hj[1:] / hs[0], "o-", color=CJ, ms=3.5, label="contact")
        ax.set_xlim(1.5, 12.5); ax.set_ylabel(r"yield $S_n/S_1$")
        ax.text(0.03, 0.87, f"{lab}  $F_0={epeak(s):.1f}$ V/nm", transform=ax.transAxes, bbox=dict(fc="w", ec="none", pad=1.0, alpha=0.85))
        ax.set_xticks(orders)
        lo = min(hs[1:].min(), hj[1:].min()) / hs[0]; hi = max(hs[1:].max(), hj[1:].max()) / hs[0]
        ax.set_ylim(lo / 12, hi * 20)
    axs[0].legend(frameon=False, loc="lower left", ncol=2, handlelength=1.8, columnspacing=1.2, borderaxespad=0.2)
    axs[1].text(0.97, 0.07, r"$n\hbar\omega>E_g$", fontsize=7, transform=axs[1].transAxes, ha="right")
    axs[1].set_xlabel("harmonic order $n$")
    fig.savefig(os.path.join(FIG, "figS2.pdf")); plt.close(fig)

# ------------------------------------------------------------------ Fig. 3
KA = 2 / 64.0 / (0.0529177 ** 2)        # eV per chain and spin -> eV per nm^2 of contact

def e2o(d):
    h = harmonics(d)
    return (h[3] + h[5]) / (h[2] + h[4])

def fig3():
    fig = plt.figure(figsize=(7.0, 1.85))
    axs = [fig.add_axes([x0, 0.215, 0.158, 0.745]) for x0 in (0.068, 0.300, 0.530, 0.835)]
    F0, Ej, Es, Fm, Fp, Fdc = [], [], [], [], [], []
    for D in DS:
        j, s = load(f"A_j_D{D}"), load(f"A_s_D{D}")
        F0.append(epeak(s)); Ej.append(j["Eabs"] * EV * KA); Es.append(s["Eabs"] * EV * KA)
        a_, b_, c_ = fields(j); Fm.append(a_); Fp.append(b_); Fdc.append(c_)
    F0 = np.array(F0)
    ax = axs[0]
    Fb, Eb = [], []
    for D in (0.5, 1.0, 1.5, 2.0):
        if have(f"B_j_D{D}"):
            b = load(f"B_j_D{D}"); Fb.append(epeak(b)); Eb.append(b["Eabs"] * EV * KA)
    ax.semilogy(Fb, Eb, "^:", color=CI, ms=3, label="contact, $U_{\\rm H}=0$")
    ax.semilogy(F0, Ej, "o-", color=CJ, ms=3, label="contact")
    ax.semilogy(F0, Es, "s--", color=CS, ms=3, mfc="w", label="insulated")
    ax.set_xlabel(r"$F_0$ (V/nm)"); ax.set_ylabel(r"absorbed energy (eV/nm$^2$)")
    ax.legend(frameon=False, loc="lower right", fontsize=6, handlelength=1.8, borderaxespad=0.15, labelspacing=0.2)
    ax.set_xlim(0.3, 1.95); ax.set_ylim(8e-6, 60); ax.text(0.05, 0.9, "(a)", transform=ax.transAxes)
    ax = axs[1]
    ax.plot([0, 2.2], [0, 2.2], "k:", lw=0.6)
    ax.plot(F0, Fp, "o-", color=CJ, ms=3, label=r"$F_{+}$")
    ax.plot(F0, Fm, "v-", color=CI, ms=3, label=r"$F_{-}$")
    ax.plot(F0, Fdc, "D-", color="#7f8c8d", ms=2.5, label=r"$F_{\rm dc}$")
    ax.set_xlim(0.3, 1.95); ax.set_ylim(0, 2.4)
    ax.set_xlabel(r"$F_0$ (V/nm)"); ax.set_ylabel("field (V/nm)")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5, handlelength=1.5, labelspacing=0.25, bbox_to_anchor=(0.0, 0.9))
    ax.text(0.05, 0.9, "(b)", transform=ax.transAxes)
    # (c) barrier height
    ax = axs[2]; ax2 = ax.twinx()
    ph, fdc, eo = [], [], []
    for p_ in (0.3, 0.5, 0.7, 1.0, 1.4, 1.67, 2.0, 2.3, 2.6):
        tag = "A_j_D5.7" if p_ == 0.7 else f"C_j_phi{p_}"
        if have(tag):
            d = load(tag); ph.append(p_); fdc.append(fields(d)[2]); eo.append(e2o(d))
    Eg = (float(d["Ec"]) - float(d["Ev"])) * EV
    ax.axvline(Eg / 2, color="k", lw=0.4, ls=":"); ax.axhline(0, color="k", lw=0.3)
    ax.plot(ph, fdc, "D-", color="#7f8c8d", ms=2.5)
    ax2.semilogy(ph, eo, "o-", color=CJ, ms=3)
    ax2.axhline(e2o(load("A_s_D5.7")), color=CJ, ls="--", lw=0.7)
    ax2.set_ylim(0.02, 3)
    ax.set_xlabel(r"$\Phi_{\rm B}$ (eV)"); ax.set_ylabel(r"$F_{\rm dc}$ (V/nm)"); ax2.set_ylabel("even/odd yield", color=CJ, labelpad=1)
    ax2.tick_params(axis="y", colors=CJ, which="both"); ax.text(0.05, 0.07, "(c)", transform=ax.transAxes)
    # (d) spacer thickness
    ax = axs[3]
    j = load("A_j_D5.7"); ds = [0.0]; En = [j["Eabs"]]; H4 = [harmonics(j)[3]]
    for dd in (0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5):
        tag = "A_s_D5.7" if dd == 1.5 else f"D_s_d{dd}"
        if have(tag):
            x = load(tag); ds.append(dd); En.append(x["Eabs"]); H4.append(harmonics(x)[3])
    ax.semilogy(ds, np.array(En) / En[0], "o-", color=CJ, ms=3, label="energy")
    ax.semilogy(ds, np.array(H4) / H4[0], "s-", color=CI, ms=3, label="4th harm.")
    ax.set_xlabel(r"spacer $d$ (nm)"); ax.set_ylabel("relative to contact"); ax.set_ylim(3e-3, 3)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", labelspacing=0.25, handlelength=1.5)
    ax.text(0.05, 0.07, "(d)", transform=ax.transAxes)
    fig.savefig(os.path.join(FIG, "figS3.pdf")); plt.close(fig)

if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    import sys
    for f in (sys.argv[1:] or ["fig1", "fig2", "fig3"]):
        globals()[f](); print("wrote", f)
