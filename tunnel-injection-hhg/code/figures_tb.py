"""Figures of the Letter from the tight-binding (Au/ZnO) simulations in ../data_tb."""
import os, glob, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from tb import DATA, EV, NM, VNM, yields

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "font.family": "serif", "mathtext.fontset": "cm",
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0, "xtick.direction": "in",
                     "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
CJ, CS, CI, CG = "#c0392b", "#2c3e50", "#2980b9", "#7f8c8d"
AU2CM2 = 3.5711e16

def L(tag):
    return dict(np.load(os.path.join(DATA, tag + ".npz"), allow_pickle=False))

def have(tag):
    return os.path.exists(os.path.join(DATA, tag + ".npz"))

# ----------------------------------------------------------------- helpers, 1D
def f1(o):
    """forward (clipped) and reverse peak fields and quasistatic component, V/nm."""
    F = o["F"]; t = o["t"] / (2 * np.pi / float(o["w"])); n = int(o["ncyc"]); m = (t > n / 2 - 1) & (t < n / 2 + 1)
    return -F.min(), F.max(), F[m].mean()

def W1(o):
    """absorbed energy, eV per nm^2 of contact."""
    return float(o["Eabs"]) * EV * 2 / float(o["Ac"]) * NM ** 2

def lin_tag(tag):
    core = tag[3:]
    key = core.rsplit("_D", 1)[0] if core[0] in "AB" else core
    return "1d_" + key + "_lin"

def Y1(tag, orders=range(1, 14)):
    """harmonic yields of the nonlinear part of the dipole acceleration (linear response subtracted)."""
    o = L(tag); P = o["P"].copy(); t = o["t"]
    lt = lin_tag(tag)
    if have(lt):
        l = L(lt); P = P - float(o["D0"]) / float(l["D0"]) * l["P"]
    return yields(t, np.gradient(P, t), float(o["w"]), orders)

def Ylin(tag):
    l = L(lin_tag(tag)); o = L(tag)
    return yields(l["t"], np.gradient(l["P"], l["t"]), float(l["w"]), [1])[0] * (float(o["D0"]) / float(l["D0"])) ** 2

def eo(Y):
    return (Y[3] + Y[5]) / (Y[2] + Y[4])

def scan(prefix, key):
    tags = [t for t in glob.glob(os.path.join(DATA, prefix + "*.npz")) if not t.endswith("_lin.npz")]
    return [os.path.basename(t)[:-4] for t in sorted(tags, key=lambda s: float(s.split(key)[-1][:-4]))]

# ----------------------------------------------------------------- Fig. 1
def schematic(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.add_patch(Rectangle((0.3, 6.0), 9.4, 0.9, fc="#d6dbdf", ec="k", lw=0.5))
    ax.add_patch(Rectangle((0.3, 4.9), 9.4, 1.1, fc="#ebedef", ec="k", lw=0.5))
    ax.add_patch(Rectangle((3.2, 6.9), 3.6, 1.0, fc="#f1c40f", ec="k", lw=0.5))
    ax.text(5.0, 7.4, "Au", ha="center", va="center", fontsize=7); ax.text(5.0, 5.2, "ZnO", ha="center", fontsize=7)
    th = np.linspace(0, 1, 30)
    for sgn, x0 in ((-1, 3.2), (1, 6.8)):
        for r_ in (0.55, 1.0, 1.5):
            ax.plot(x0 + sgn * r_ * (1 - np.cos(th * np.pi / 2)) * 1.2, 6.9 - r_ * np.sin(th * np.pi / 2), color=CJ, lw=0.6)
    ax.add_patch(FancyArrowPatch((6.8, 6.9), (8.55, 5.45), arrowstyle="->", mutation_scale=7, color="k", lw=0.8, connectionstyle="arc3,rad=0.28"))
    ax.text(8.65, 5.15, "$z$", fontsize=8)
    ax.text(5.0, 8.6, r"$E_{\rm L}(t)$", ha="center", fontsize=8)
    ax.add_patch(FancyArrowPatch((3.9, 8.25), (6.1, 8.25), arrowstyle="<->", mutation_scale=6, color="k", lw=0.7))
    y0 = 0.4
    ax.add_patch(Rectangle((0.6, y0), 2.6, 2.0, fc="#f9e79f", ec="none"))
    ax.plot([0.6, 3.2], [y0 + 2.0, y0 + 2.0], "k", lw=0.8); ax.text(0.7, y0 + 2.15, r"$E_{\rm F}$", fontsize=7)
    zz = np.linspace(0, 1, 60); bend = 1.5 * (1 - np.exp(-zz / 0.35)) * 0.6; xs = 3.2 + 6.2 * zz
    ax.plot(xs, y0 + 2.55 - bend, color=CS, lw=0.9); ax.plot(xs, y0 + 0.75 - bend, color=CS, lw=0.9)
    ax.plot([3.2, 3.2], [y0, y0 + 3.3], "k", lw=0.5)
    ax.text(9.5, y0 + 2.55 - bend[-1] + 0.12, "CB", fontsize=6.5, ha="right"); ax.text(9.5, y0 + 0.75 - bend[-1] - 0.42, "VB", fontsize=6.5, ha="right")
    ax.annotate("", xy=(3.28, y0 + 2.55), xytext=(3.28, y0 + 2.0), arrowprops=dict(arrowstyle="<->", lw=0.5, shrinkA=0, shrinkB=0))
    ax.text(3.4, y0 + 2.75, r"$\Phi_{\rm B}$", fontsize=7)
    ax.add_patch(FancyArrowPatch((2.6, y0 + 1.9), (4.6, y0 + 1.9), arrowstyle="->", mutation_scale=7, color=CJ, lw=1.0))
    ax.text(4.75, y0 + 1.72, r"$e^-$", color=CJ, fontsize=7); ax.text(1.9, y0 + 0.9, "metal", ha="center", fontsize=7)
    ax.text(0.2, 9.5, "(a)", fontsize=8)

def fig1():
    fig = plt.figure(figsize=(7.0, 1.95))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.25, 1.1], wspace=0.36, hspace=0.12, left=0.01, right=0.99, top=0.97, bottom=0.185)
    schematic(fig.add_subplot(gs[:, 0]))
    c, i = L("1d_G_c_D5.5"), L("1d_G_i_D5.5")
    T = 2 * np.pi / float(c["w"]); z = c["z"]; b = float(c["b"]); Ac = float(c["Ac"])
    axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, 1], sharex=axb)
    axb.plot(i["t"] / T, i["F"], color=CS, ls="--", label="insulated"); axb.plot(c["t"] / T, c["F"], color=CJ, label="contact")
    axb.axhline(0, color="k", lw=0.3); axb.set_ylabel(r"$F$ (V/nm)"); axb.set_ylim(-1.45, 2.35)
    axb.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.6, columnspacing=1.0); plt.setp(axb.get_xticklabels(), visible=False)
    w2 = (z >= 2 * b).astype(float)
    dq = ((c["snaps"] - i["snaps"]) * w2).sum(1) * 2 / Ac * AU2CM2 / 1e12
    axc.plot(c["snaps_t"] / T, dq, color=CJ); axc.axhline(0, color="k", lw=0.3)
    axc.set_ylabel(r"$\Delta\sigma$ ($10^{12}$ cm$^{-2}$)"); axc.set_xlabel("time (optical cycles)"); axc.set_xlim(0, 5)
    axb.text(0.985, 0.86, "(b)", transform=axb.transAxes, ha="right"); axc.text(0.985, 0.82, "(c)", transform=axc.transAxes, ha="right")
    axd = fig.add_subplot(gs[:, 2])
    nm = int(c["nm_layers"]); zl = z[nm::2]
    st = c["snaps_t"] / T
    for tt, col, lab in ((2.0, CJ, r"$t=2T$"), (2.5, CI, r"$t=2.5T$")):
        k = np.argmin(abs(st - tt)); d = (c["snaps"][k] - i["snaps"][k])[nm:]
        dl = d[0::2] + d[1::2]                                   # s + p of each layer
        dn = np.convolve(dl, np.ones(3) / 3, mode="same") * 2 / (Ac * b) * NM ** 3
        axd.plot(zl / NM, dn * 1e3, color=col, label=lab)
    axd.axvline(0, color="k", lw=0.4); axd.axhline(0, color="k", lw=0.3)
    axd.set_xlim(-0.3, 10); axd.set_ylim(-28, 62); axd.set_xlabel(r"$z$ (nm)"); axd.set_ylabel(r"$\Delta(\delta n)$ ($10^{-3}$ nm$^{-3}$)")
    axd.legend(frameon=False, loc="upper right"); axd.text(0.9, 0.06, "(d)", transform=axd.transAxes)
    fig.savefig(os.path.join(FIG, "fig1.pdf")); plt.close(fig)

# ----------------------------------------------------------------- Fig. 2 (2D strip array)
def nl2(hi, lo, key):
    h, l = L(hi), L(lo)
    return h["t"], h[key] - float(h["F0"]) / float(l["F0"]) * l[key], float(h["w"])

def fig2(F0="0.75", tsnap=3.0):
    if not have(f"G1_F{F0}_right"):
        F0 = "0.5"
    fig = plt.figure(figsize=(7.0, 2.0))
    c, i, r = L(f"G1_F{F0}_both"), L(f"G1_F{F0}_none"), L(f"G1_F{F0}_right")
    T = 2 * np.pi / float(c["w"]); b = float(c["b"]); Nx, Nz, Wm, Hm = int(c["Nx"]), int(c["Nz"]), int(c["Wm"]), int(c["Hm"])
    Ly = float(c["Ly"]); i0 = (Nx - Wm) // 2
    # (a) induced electron density of the one-sided contact minus insulated
    ax = fig.add_axes([0.055, 0.2, 0.36, 0.62])
    k = np.argmin(abs(r["snaps_t"] / T - tsnap))
    d = (r["snaps"][k] - i["snaps"][np.argmin(abs(i["snaps_t"] / T - tsnap))]) * 2 / (Ly * b * b) * NM ** 3   # electrons per nm^3
    ix = np.rint(r["sx"] / b).astype(int); iz = np.rint(r["sz"] / b).astype(int)
    G = np.full((Nz + Hm, Nx), np.nan); G[iz + Hm, ix] = d
    ker = np.array([[0.25, 0.5, 0.25]]); 
    Gc = G.copy(); Gc[np.isnan(Gc)] = 0.0
    from scipy.ndimage import uniform_filter
    Gs = uniform_filter(Gc, size=2, mode="wrap")
    ext = [-0.5 * b / NM, (Nx - 0.5) * b / NM, (Nz - 0.5) * b / NM, -(Hm + 0.5) * b / NM]
    vmax = 0.06
    im = ax.imshow(Gs * 1e3, cmap="RdBu_r", vmin=-vmax * 1e3, vmax=vmax * 1e3, extent=ext, aspect="equal", interpolation="nearest")
    ax.add_patch(Rectangle(((i0 - 0.5) * b / NM, -(Hm + 0.5) * b / NM), Wm * b / NM, Hm * b / NM, fc="#f1c40f", ec="k", lw=0.6))
    xm = (i0 + Wm / 2 - 0.5) * b / NM
    ax.plot([(i0 - 0.5) * b / NM, xm], [-0.5 * b / NM] * 2, color="k", lw=2.0, solid_capstyle="butt")
    ax.set_ylim((Nz - 0.5) * b / NM, -(Hm + 2.5) * b / NM); ax.set_xlabel(r"$x$ (nm)"); ax.set_ylabel(r"$z$ (nm)")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.06, pad=0.03, location="top", aspect=40, extend="both")
    cb.set_label(r"$\Delta(\delta n)$ ($10^{-3}$ nm$^{-3}$)", fontsize=7, labelpad=2); cb.ax.tick_params(labelsize=6, pad=1)
    ax.text(0.015, 0.06, "(a)", transform=ax.transAxes)
    # (b) normal field under the contacted edge and charge transferred into the crystal
    axb = fig.add_axes([0.495, 0.60, 0.195, 0.345]); axc = fig.add_axes([0.495, 0.2, 0.195, 0.345])
    axb.plot(i["t"] / T, i["F"][:, 3], color=CS, ls="--", label="insulated"); axb.plot(r["t"] / T, r["F"][:, 3], color=CJ, label="contact")
    axb.axhline(0, color="k", lw=0.3); axb.set_ylabel(r"$F_z$ (V/nm)"); plt.setp(axb.get_xticklabels(), visible=False); axb.set_xlim(0, 5)
    lim = 1.5 * np.abs(i["F"][:, 3]).max(); axb.set_ylim(-lim, 1.75 * lim)
    axb.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.4, columnspacing=0.8, fontsize=6.5, borderaxespad=0.1)
    def cyc(o):
        q = (o["QR"] + o["QL"]) * 2 / Ly * NM; t = o["t"]; n = int(round(T / (t[1] - t[0])))
        return t / T, q, np.convolve(q, np.ones(n) / n, mode="same"), n
    for o, col, ls in ((i, CS, "--"), (r, CJ, "-")):
        tt, q, qa, n = cyc(o); axc.plot(tt, q, color=col, lw=0.4, alpha=0.45, ls=ls); axc.plot(tt[n // 2:-n // 2], qa[n // 2:-n // 2], color=col, ls=ls)
    axc.axhline(0, color="k", lw=0.3)
    axc.set_ylabel(r"$\Delta N$ (nm$^{-1}$)"); axc.set_xlabel("time (optical cycles)"); axc.set_xlim(0, 5)
    axb.text(0.97, 0.08, "(b)", transform=axb.transAxes, ha="right"); axc.text(0.97, 0.08, "(c)", transform=axc.transAxes, ha="right")
    # (d) nonlinear in-plane emission
    axd = fig.add_axes([0.775, 0.2, 0.215, 0.745])
    lo = "G1_F0.025_"
    l0 = L(lo + "none"); y1 = yields(l0["t"], l0["Jx"], float(l0["w"]), [1])[0] * (float(F0) / 0.025) ** 2
    orders = np.arange(2, 10)
    for tag, col, mk, ls, lab, kw in (("none", CS, "s", "--", "insulated", dict(mfc="w")), ("right", CJ, "o", "-", "one end in contact", {}), ("both", CI, "^", ":", "both ends", {})):
        t, J, w = nl2(f"G1_F{F0}_{tag}", lo + tag, "Jx")
        Y = yields(t, J, w, range(1, 10))
        axd.semilogy(orders, Y[1:] / y1, mk + ls, color=col, ms=3.5, label=lab, **kw)
    axd.set_xlabel("harmonic order $n$"); axd.set_ylabel(r"in-plane yield $S_n/S_1$"); axd.set_xticks(orders)
    axd.legend(frameon=False, fontsize=6.5, loc="upper right", labelspacing=0.2, handlelength=1.8, borderaxespad=0.2)
    axd.text(0.05, 0.06, "(d)", transform=axd.transAxes)
    fig.savefig(os.path.join(FIG, "fig2.pdf")); plt.close(fig)

# ----------------------------------------------------------------- Fig. 3 (1D scans)
def fig3():
    fig = plt.figure(figsize=(7.0, 1.7))
    axs = [fig.add_axes([x0, 0.235, 0.158, 0.725]) for x0 in (0.068, 0.300, 0.530, 0.835)]
    ct = scan("1d_A_c_D", "_D"); F0, Wc, Wi, Fm, Fp, Fd, Fmi = [], [], [], [], [], [], []
    for tag in ct:
        ti = tag.replace("_A_c_", "_A_i_")
        if not have(ti): continue
        c, i = L(tag), L(ti); a_, b_, d_ = f1(c); ai, bi, _ = f1(i)
        F0.append(bi); Wc.append(W1(c)); Wi.append(W1(i)); Fm.append(a_); Fp.append(b_); Fd.append(d_); Fmi.append(ai)
    F0 = np.array(F0); sel = F0 < 2.7
    ax = axs[0]
    Fb, Wb = [], []
    for tag in scan("1d_B_c_D", "_D"):
        o = L(tag); Fb.append(f1(o)[1]); Wb.append(W1(o))
    ax.semilogy(Fb, Wb, "^:", color=CI, ms=3, label=r"contact, $U_{\rm H}=0$")
    ax.semilogy(F0[sel], np.array(Wc)[sel], "o-", color=CJ, ms=3, label="contact")
    ax.semilogy(F0[sel], np.array(Wi)[sel], "s--", color=CS, ms=3, mfc="w", label="insulated")
    ax.set_xlabel(r"$F_0$ (V/nm)"); ax.set_ylabel(r"energy (eV/nm$^2$)")
    ax.legend(frameon=False, loc="lower right", fontsize=6, handlelength=1.8, borderaxespad=0.15, labelspacing=0.2)
    ax.set_xlim(0.25, 2.7); ax.set_ylim(2e-7, 60); ax.text(0.05, 0.9, "(a)", transform=ax.transAxes)
    ax = axs[1]
    ax.plot([0, 3], [0, 3], "k:", lw=0.6)
    ax.plot(F0[sel], np.array(Fp)[sel], "o-", color=CJ, ms=3, label=r"$F_{+}$")
    ax.plot(F0[sel], np.array(Fm)[sel], "v-", color=CI, ms=3, label=r"$F_{-}$")
    ax.plot(F0[sel], np.array(Fmi)[sel], ":", color=CI, lw=0.8)
    ax.plot(F0[sel], np.array(Fd)[sel], "D-", color=CG, ms=2.5, label=r"$F_{\rm dc}$")
    ax.set_xlim(0.25, 2.7); ax.set_ylim(0, 3.3); ax.set_xlabel(r"$F_0$ (V/nm)"); ax.set_ylabel("field (V/nm)")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5, handlelength=1.5, labelspacing=0.25, bbox_to_anchor=(0.0, 0.9))
    ax.text(0.05, 0.9, "(b)", transform=ax.transAxes)
    ax = axs[2]; ax2 = ax.twinx()
    ph, fd, e2 = [], [], []
    for tag in scan("1d_C_c_phi", "_phi") + ["1d_A_c_D5.5"]:
        o = L(tag); ph.append(0.7 if "A_c" in tag else float(tag.split("_phi")[-1])); fd.append(f1(o)[2]); e2.append(Y1(tag)[1] / Ylin(tag))
    k = np.argsort(ph); ph, fd, e2 = np.array(ph)[k], np.array(fd)[k], np.array(e2)[k]
    ax.axvline(1.65, color="k", lw=0.4, ls=":"); ax.axhline(0, color="k", lw=0.3)
    ax.plot(ph, fd, "D-", color=CG, ms=2.5); ax2.semilogy(ph, e2, "o-", color=CJ, ms=3)
    ax2.axhline(Y1("1d_A_i_D5.5")[1] / Ylin("1d_A_i_D5.5"), color=CJ, ls="--", lw=0.7)
    ax.set_xlabel(r"$\Phi_{\rm B}$ (eV)"); ax.set_ylabel(r"$F_{\rm dc}$ (V/nm)"); ax2.set_ylabel(r"second harmonic $S_2/S_1$", color=CJ, labelpad=1)
    ax2.tick_params(axis="y", colors=CJ, which="both"); ax.text(0.05, 0.07, "(c)", transform=ax.transAxes)
    ax = axs[3]
    ti, fdt, wt = [], [], []
    for tag in scan("1d_D_c_t", "_t") + ["1d_A_c_D5.5"]:
        o = L(tag); ti.append(1.5 if "A_c" in tag else float(tag.split("_t")[-1])); fdt.append(f1(o)[2]); wt.append(W1(o))
    k = np.argsort(ti); ti, fdt, wt = np.array(ti)[k], np.array(fdt)[k], np.array(wt)[k]
    ref = L("1d_A_i_D5.5")
    ax.semilogx(ti, fdt / fdt[ti == 1.5], "D-", color=CG, ms=2.5, label=r"$F_{\rm dc}$")
    ax.semilogx(ti, wt / wt[ti == 1.5], "o-", color=CJ, ms=3, label="energy")
    ax.axhline(W1(ref) / wt[ti == 1.5][0], color=CJ, ls="--", lw=0.7)
    ax.set_xlabel(r"$t_c$ (eV)"); ax.set_ylabel("relative to $t_c=1.5$ eV"); ax.set_ylim(-0.1, 1.25)
    ax.legend(frameon=False, fontsize=6.5, loc="upper left", labelspacing=0.25, handlelength=1.5, bbox_to_anchor=(0.0, 0.88))
    ax.text(0.05, 0.9, "(d)", transform=ax.transAxes)
    fig.savefig(os.path.join(FIG, "fig3.pdf")); plt.close(fig)

# ----------------------------------------------------------------- Fig. S1 (1D harmonic yields)
def figS1():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2)); fig.subplots_adjust(left=0.075, right=0.99, bottom=0.2, top=0.95, wspace=0.36)
    orders = np.arange(2, 12)
    for ax, D, lab in ((axs[0], "4.6", "(a)"), (axs[1], "6.4", "(b)")):
        tc, ti = f"1d_A_c_D{D}", f"1d_A_i_D{D}"; y1 = Ylin(ti)
        ax.semilogy(orders, Y1(ti)[1:11] / y1, "s--", color=CS, ms=3.5, mfc="w", label="insulated")
        ax.semilogy(orders, Y1(tc)[1:11] / y1, "o-", color=CJ, ms=3.5, label="contact")
        ax.axvspan(5.32, 11.6, color="0.92", zorder=0); ax.set_xlim(1.6, 11.4); ax.set_xticks(orders)
        ax.set_xlabel("harmonic order $n$"); ax.set_ylabel("$S_n/S_1$")
        ax.text(0.05, 0.06, lab + f" $F_0={f1(L(ti))[1]:.1f}$ V/nm", transform=ax.transAxes)
    axs[0].set_ylim(2e-8, 6e-3); axs[0].legend(frameon=False, loc="upper left", ncol=2, columnspacing=1.0)
    ax = axs[2]; F0 = []; Yc = []; Yi = []
    for tag in scan("1d_A_c_D", "_D"):
        ti = tag.replace("_A_c_", "_A_i_"); F0.append(f1(L(ti))[1]); y1 = Ylin(ti); Yc.append(Y1(tag) / y1); Yi.append(Y1(ti) / y1)
    F0 = np.array(F0); Yc = np.array(Yc); Yi = np.array(Yi); sel = F0 < 2.7
    for n, col, mk in ((2, CJ, "o"), (4, CI, "^")):
        ax.semilogy(F0[sel], Yc[sel, n - 1], mk + "-", color=col, ms=3.5, label=f"$n={n}$")
        ax.semilogy(F0[sel], Yi[sel, n - 1], mk + "--", color=col, ms=3.5, mfc="w")
    ax.set_xlabel(r"$F_0$ (V/nm)"); ax.set_ylabel("$S_n/S_1$"); ax.legend(frameon=False, loc="lower right"); ax.text(0.05, 0.9, "(c)", transform=ax.transAxes)
    fig.savefig(os.path.join(FIG, "figS_harm.pdf")); plt.close(fig)

if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    for f in (sys.argv[1:] or ["fig1", "fig2", "fig3"]):
        globals()[f](); print("wrote", f)
