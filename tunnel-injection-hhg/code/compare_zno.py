"""Hot-spot-averaged response for the antennas of Imasaka et al. [Opt. Express 26, 21364 (2018)].

Geometry: gold bars 570 nm x 310 nm x 100 nm on ZnO, driver 2 um. The shape of the near field around the
contact edge is taken from the quasi-static solution (nearfield.py); its magnitude is fixed by the reported
enhancement of |E_x| 2 nm below the surface. The local response (holes, absorbed energy, harmonic yields per
unit area) is taken from the chain model: contacted junction under the metal, insulated junction elsewhere."""
import os, sys
import numpy as np
from nearfield import solve, fields
from figures_tb import L, f1, W1, scan, Y1, Ylin, DATA
from carriers_tb import carriers

E_PER_SQRT_TW = 2.745          # vacuum field amplitude (V/nm) at 1 TW/cm^2

def near_field(Lbar=570.0, h=100.0, eps=3.7, cache=os.path.join(DATA, "nearfield_570x100.npz")):
    if os.path.exists(cache):
        d = np.load(cache); return d["s"], d["Fz"], d["Fabs"], float(d["Exmax2"]), d["zprof"], d["eprof"]
    x, z, phi = solve(Lbar, h, eps, hmin=0.25)
    Ex, Ez = fields(x, z, phi)
    s = x - Lbar / 2
    jz = (z >= 0.5 - 1e-6) & (z <= 2.0 + 1e-6)
    Fz = Ez[:, jz].mean(1); Fabs = np.hypot(Ex[:, jz], Ez[:, jz]).mean(1)
    j2 = np.argmin(abs(z - 2.0)); Exmax2 = np.abs(Ex[:, j2]).max()
    i1 = np.argmin(abs(s + 1.0)); zp = z[z >= 0]; ep = Ez[i1, z >= 0] / Ez[i1, jz].mean()      # depth profile 1 nm inside the edge
    np.savez(cache, s=s, Fz=Fz, Fabs=Fabs, Exmax2=Exmax2, zprof=zp, eprof=ep)
    return s, Fz, Fabs, float(Exmax2), zp, ep

def local_tables(prefix="1d_A", key="_D"):
    """chain-model results versus local field: dict name -> (F0, contact, insulated)."""
    F0, rows_c, rows_i = [], [], []
    for tag in scan(prefix + "_c" + key[:2], key):
        ti = tag.replace("_c_", "_i_"); c, i = L(tag), L(ti); F = f1(i)[1]
        y1 = Ylin(ti) / F ** 2                                   # linear yield per unit field^2
        hc, ec = carriers(c); hi, ei = carriers(i)
        Yc, Yi = Y1(tag), Y1(ti)
        F0.append(F); rows_c.append([hc, W1(c)] + [Yc[n - 1] / y1 for n in (2, 3, 4, 5, 6, 7, 9)])
        rows_i.append([hi, W1(i)] + [Yi[n - 1] / y1 for n in (2, 3, 4, 5, 6, 7, 9)])
    return np.array(F0), np.array(rows_c), np.array(rows_i), ["holes", "W", "S2", "S3", "S4", "S5", "S6", "S7", "S9"]

def interp_loglog(F0, y, F, slope_lo=None, fit_lo=(0, 1)):
    """log-log interpolation; below the first point a power law (slope_lo in field, or from two points)."""
    lf, ly = np.log(F0), np.log(np.maximum(y, 1e-300))
    out = np.interp(np.log(F), lf, ly)
    a, b = fit_lo
    sl = slope_lo if slope_lo is not None else (ly[b] - ly[a]) / (lf[b] - lf[a])
    lo = F < F0[a]
    out[lo] = ly[a] + sl * (np.log(F[lo]) - lf[a])
    return np.exp(out)

def hot_spot(I_TW, enh=9.0, ref="vacuum", gap=150.0, contact=True, tabs=None, floor_fix=True):
    """response integrated over the hot spot per unit length of the edge, for incident intensity I (TW/cm^2)."""
    s, Fz, Fabs, Exmax2, _, _ = near_field()
    F0, Rc, Ri, names = tabs
    Ein = E_PER_SQRT_TW * np.sqrt(I_TW) * (1.0 if ref == "vacuum" else 2 / (1 + np.sqrt(3.7)))
    scale = enh * Ein / Exmax2                                    # V/nm per unit of the quasi-static solution
    under = (s <= 0) & (s >= -285.0); outside = (s > 0) & (s <= gap)
    res = []
    for k in range(Rc.shape[1]):
        # the insulated tables are limited by a numerical floor at the lowest fields: extrapolate from 0.82-1.0 V/nm
        yi = lambda F: interp_loglog(F0[3:], Ri[3:, k], F) if floor_fix else interp_loglog(F0, Ri[:, k], F)
        yc = lambda F: interp_loglog(F0, Rc[:, k], F)
        fu = (yc if contact else yi)(np.maximum(np.abs(Fz[under]) * scale, 1e-6))
        fo = yi(np.maximum(Fabs[outside] * scale, 1e-6))
        res.append(np.trapezoid(fu, s[under]) + np.trapezoid(fo, s[outside]))
    return np.array(res), scale * Exmax2

# measured exponents with antennas, 0.012-0.1 TW/cm^2, read off Fig. 8 of Ashihara and Imasaka,
# Rev. Laser Eng. 48, 174 (2020), which reproduces the data of the Optics Express paper (uncertainty about 0.3)
MEASURED = {"S5": 3.4, "S7": 3.2, "S9": 3.0, "holes": 2.7}
MULTIPHOTON = {"S5": 5, "S7": 7, "S9": 9}

def exponents(tabs, enh, ref="substrate", I=None):
    """power-law exponents of the hot-spot yields over the measured range, contact and insulated."""
    I = np.geomspace(0.012, 0.1, 12) if I is None else I; names = tabs[3]
    Rc = np.array([hot_spot(i, enh, ref, contact=True, tabs=tabs)[0] for i in I]); Ri = np.array([hot_spot(i, enh, ref, contact=False, tabs=tabs)[0] for i in I])
    sl = lambda R, n: np.polyfit(np.log(I), np.log(R[:, names.index(n)]), 1)[0]
    return {n: (sl(Rc, n), sl(Ri, n)) for n in ("S5", "S7", "S9", "holes", "W")}

def figure(prefix="1d_P", ref="substrate"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from figures_tb import FIG, CJ, CS, CI, CG
    tabs = local_tables(prefix); names = tabs[3]
    s, Fz, Fabs, Exmax2, zp, ep = near_field()
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3)); fig.subplots_adjust(left=0.065, right=0.985, bottom=0.19, top=0.95, wspace=0.33)
    ax = axs[0]
    m = (s >= -60) & (s <= 60)
    ax.plot(s[m], Fabs[m] / Exmax2, color=CS, label=r"$|{\bf E}|$")
    ax.plot(s[m & (s <= 0)], Fz[m & (s <= 0)] / Exmax2, color=CJ, ls="--", label=r"$E_z$")
    ax.axvline(0, color="k", lw=0.4, ls=":"); ax.set_xlabel("distance from the contact edge (nm)"); ax.set_ylabel("field / peak $|E_x|$ at 2 nm depth")
    ax.set_ylim(0, 1.6); ax.legend(frameon=False, loc="upper left"); ax.text(0.9, 0.9, "(a)", transform=ax.transAxes)
    ax.text(-57, 0.08, "under the metal", fontsize=6.5); ax.text(12, 0.08, "free surface", fontsize=6.5)
    ax = axs[1]
    enhs = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.5, 9.0, 10.5]); ex = [exponents(tabs, e, ref) for e in enhs]
    for name, col, lab in (("holes", CJ, "fluorescence (holes)"), ("S7", CI, "7th harmonic")):
        ax.plot(enhs, [e[name][0] for e in ex], "o-", color=col, ms=3, label=lab)
        ax.plot(enhs, [e[name][1] for e in ex], "s--", color=col, ms=3, mfc="w")
        ax.axhspan(MEASURED[name] - 0.3, MEASURED[name] + 0.3, color=col, alpha=0.15, lw=0)
    ax.set_xlabel("peak field enhancement"); ax.set_ylabel("intensity exponent"); ax.set_ylim(1.2, 5.0)
    ax.legend(frameon=False, loc="upper right"); ax.text(0.05, 0.07, "(b)", transform=ax.transAxes)
    ax = axs[2]
    I = np.geomspace(4e-4, 0.1, 40)
    for enh, ls in ((4.0, "-"), (9.0, "--")):
        Rc = np.array([hot_spot(i, enh, ref, contact=True, tabs=tabs)[0] for i in I]); Ri = np.array([hot_spot(i, enh, ref, contact=False, tabs=tabs)[0] for i in I])
        for name, col, lab in (("W", CJ, "absorbed energy"), ("S2", CI, "second harmonic")):
            k = names.index(name); ax.loglog(I, Rc[:, k] / Ri[:, k], color=col, ls=ls, label=lab if enh == 4.0 else None)
    ax.axhline(1, color="k", lw=0.4); ax.axvspan(0.012, 0.1, color="0.92", zorder=0); ax.set_xlim(4e-4, 0.13); ax.set_ylim(0.5, 3e4)
    ax.set_xlabel(r"incident intensity (TW/cm$^2$)"); ax.set_ylabel("contact / insulated"); ax.legend(frameon=False, loc="upper right")
    ax.text(0.05, 0.07, "(c)", transform=ax.transAxes)
    fig.savefig(os.path.join(FIG, "figS_zno.pdf")); plt.close(fig)

def tables(out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")):
    rows = []
    for prefix, lab in (("1d_P", "computed profile"), ("1d_A", r"$\ell=8$ nm")):
        tabs = local_tables(prefix)
        for enh in (3.0, 4.0, 5.0, 9.0):
            e = exponents(tabs, enh)
            rows.append(f"{lab} & {enh:.0f} & " + " & ".join(f"{e[n][0]:.1f} ({e[n][1]:.1f})" for n in ("S5", "S7", "S9", "holes")) + r" \\")
        rows.append(r"\hline")
    rows.append(r"Measured & & " + " & ".join(f"{MEASURED[n]:.1f}" for n in ("S5", "S7", "S9", "holes")) + r" \\")
    rows.append(r"Multiphoton law & & 5 & 7 & 9 & \\")
    lines = [r"\begin{tabular}{llcccc}", r"\hline\hline", r"Near-field profile & Enhancement & 5th harmonic & 7th harmonic & 9th harmonic & Fluorescence \\", r"\hline"] + rows + [r"\hline\hline", r"\end{tabular}"]
    open(os.path.join(out_dir, "tab_zno.tex"), "w").write("\n".join(lines) + "\n")

if __name__ == "__main__":
    s, Fz, Fabs, Exmax2, zp, ep = near_field()
    print(f"quasi-static: max |Ex|/E0 at 2 nm depth = {Exmax2:.2f}")
    print("   depth profile 1 nm inside the edge (z nm, e):", [(round(float(zp[np.argmin(abs(zp - d))]), 1), round(float(ep[np.argmin(abs(zp - d))]), 2)) for d in (0.5, 1, 2, 4, 8, 16, 32, 64)])
    figure(); tables()
    tabs = local_tables("1d_P"); names = tabs[3]
    I = np.geomspace(4e-4, 0.1, 60)
    for enh in (4.0, 9.0):
        Rc = np.array([hot_spot(i, enh, "substrate", contact=True, tabs=tabs)[0] for i in I]); Ri = np.array([hot_spot(i, enh, "substrate", contact=False, tabs=tabs)[0] for i in I])
        pk = [hot_spot(i, enh, "substrate", contact=True, tabs=tabs)[1] for i in (0.012, 0.1)]
        print(f"enhancement {enh}: peak local field {pk[0]:.2f}-{pk[1]:.2f} V/nm over 0.012-0.1 TW/cm2; exponents {exponents(tabs, enh)}")
        m = I >= 0.012
        for name in ("W", "holes", "S2", "S4"):
            k = names.index(name); r = Rc[:, k] / Ri[:, k]
            print(f"   {name:5s} contact/insulated: {r[m].min():.1f}-{r[m].max():.1f} in the measured range; > 10 below {I[r > 10].max() if (r > 10).any() else float('nan'):.4f} TW/cm2; at 0.012: {np.interp(0.012, I, r):.1f}, at 0.03: {np.interp(0.03, I, r):.1f}")
