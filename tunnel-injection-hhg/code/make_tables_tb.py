"""LaTeX tables for the tight-binding (Au/ZnO) results."""
import os, glob
import numpy as np
from tb import DATA, EV, NM, VNM, yields
from figures_tb import L, have, f1, W1, Y1, Ylin, eo, scan

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")

def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"

def write(name, header, rows):
    ncol = max(r.count("&") for r in rows) + 1
    lines = [r"\begin{tabular}{" + "l" + "c" * (ncol - 1) + "}", r"\hline\hline"] + header + [r"\hline"] + rows + [r"\hline\hline", r"\end{tabular}"]
    open(os.path.join(OUT, name), "w").write("\n".join(lines) + "\n")

# 1D field scan
rows = []
for tag in scan("1d_A_c_D", "_D"):
    ti = tag.replace("_A_c_", "_A_i_")
    if not have(ti): continue
    c, i = L(tag), L(ti); fm, fp, fd = f1(c); fmi, fpi, _ = f1(i); Yc, Yi = Y1(tag), Y1(ti)
    rows.append(f"{float(c['D0']):.1f} & {fpi:.2f} & {fmi:.2f} & {fm:.2f} & {fp:.2f} & {fd:.2f} & {W1(c):.4f} & {sci(W1(i))} & {eo(Yc):.2f} & {eo(Yi):.3f} \\\\")
write("tab_tb_field.tex", [r"$D_0$ & $F_0$ & $F_-^{\rm ins}$ & $F_-$ & $F_+$ & $F_{\rm dc}$ & $W_{\rm contact}$ & $W_{\rm insulated}$ & \multicolumn{2}{c}{even/odd} \\",
                            r"\multicolumn{6}{c}{(V/nm)} & \multicolumn{2}{c}{(eV/nm$^2$)} & contact & insulated \\"], rows)

def row1(tag, label):
    c = L(tag); fm, fp, fd = f1(c); Y = Y1(tag); y1 = Ylin(tag)
    return f"{label} & {fm:.2f} & {fp:.2f} & ${fd:+.2f}$ & {W1(c):.3f} & {eo(Y):.2f} & {sci(Y[3]/y1)} & {sci(Y[4]/y1)} & {sci(Y[8]/y1)} \\\\"
H1 = [r"Run & $F_-$ & $F_+$ & $F_{\rm dc}$ & $W$ & even/odd & $S_4/S_1$ & $S_5/S_1$ & $S_9/S_1$ \\", r" & \multicolumn{3}{c}{(V/nm)} & (eV/nm$^2$) & & & & \\"]
rows = [row1(t, f"$\\Phi_{{\\rm B}}={0.7 if 'A_c' in t else float(t.split('_phi')[-1]):.2f}$ eV") for t in sorted(scan("1d_C_c_phi", "_phi") + ["1d_A_c_D5.5"], key=lambda t: 0.7 if "A_c" in t else float(t.split("_phi")[-1]))]
write("tab_tb_phi.tex", H1, rows)
rows = [row1(t, f"$t_c={1.5 if 'A_c' in t else float(t.split('_t')[-1]):.2f}$ eV") for t in sorted(scan("1d_D_c_t", "_t") + ["1d_A_c_D5.5"], key=lambda t: 1.5 if "A_c" in t else float(t.split("_t")[-1]))]
rows.append(row1("1d_A_i_D5.5", "$t_c=0$ (insulated)"))
write("tab_tb_tc.tex", H1, rows)
rows = []
for tag, lab in (("1d_A_c_D5.5", "Reference"), ("1d_E_dt01_c", r"$\Delta t=0.1$"), ("1d_E_long_c", "Crystal 260 layers"), ("1d_E_metal_c", "Metal 140 layers"),
                 (None, None), ("1d_F_l4_c", r"Contact, $\ell=4$ nm"), ("1d_F_l4_i", r"Insulated, $\ell=4$ nm"), ("1d_F_l20_c", r"Contact, $\ell=20$ nm"), ("1d_F_l20_i", r"Insulated, $\ell=20$ nm"),
                 (None, None), ("1d_W_3200_c", r"Contact, 3.2 $\mu$m"), ("1d_W_3200_i", r"Insulated, 3.2 $\mu$m"),
                 (None, None), ("1d_I_8cyc_c", "Contact, 8 cycles"), ("1d_I_8cyc_i", "Insulated, 8 cycles")):
    if tag is None: rows.append(r"\hline")
    elif have(tag): rows.append(row1(tag, lab))
write("tab_tb_extra.tex", H1, rows)

# 2D
def nl(hi, lo, key):
    h, l = L(hi), L(lo); return h["t"], h[key] - float(h["F0"]) / float(l["F0"]) * l[key], float(h["w"])

def rows2d(pre, F0, confs, label):
    out = []
    for c, lab in confs:
        hi, lo = f"{pre}_F{F0}_{c}", f"{pre}_F0.025_{c}"
        if not (have(hi) and have(lo)): continue
        ref = f"{pre}_F0.025_none" if have(f"{pre}_F0.025_none") else f"{pre}_F0.025_right"   # common normalization
        l0 = L(ref); y1 = yields(l0["t"], l0["Jx"], float(l0["w"]), [1])[0] * (float(F0) / 0.025) ** 2
        o = L(hi); t, J, w = nl(hi, lo, "Jx"); Y = yields(t, J, w, range(1, 8))
        T = 2 * np.pi / w; n = int(o["ncyc"]); m = (o["t"] / T > n / 2 - 1) & (o["t"] / T < n / 2 + 1); F = o["F"]
        out.append(f"{label} & {F0} & {lab} & {np.abs(F[:, 0]).max():.2f} & {F[:, 3].max():.2f} & {-F[:, 3].min():.2f} & ${F[m, 3].mean():+.2f}$ & {float(o['Eabs'])*EV*2/float(o['Ly'])*NM:.3f} & "
                   + " & ".join(sci(Y[k - 1] / y1) for k in (2, 3, 4, 5)) + r" \\")
    return out

C3 = (("none", "insulated"), ("right", "one end"), ("both", "both ends"))
rows = []
for pre, F0, confs, label in (("G1", "0.5", C3, "Reference"), ("G1", "0.75", C3, "Reference"), ("G2", "0.75", C3[:2], "Slab of 32 layers"), ("G1c8", "0.75", C3[1:], "Eight cycles")):
    r = rows2d(pre, F0, confs, label)
    if r: rows += r + [r"\hline"]
write("tab_tb_2d.tex", [r"Run & $F_{\rm m}$ & Contact & $|F_x|^{\rm out}$ & $F_{z,+}$ & $F_{z,-}$ & $F_{z,{\rm dc}}$ & $W$ & $S_2/S_1$ & $S_3/S_1$ & $S_4/S_1$ & $S_5/S_1$ \\",
                         r" & (V/nm) & & \multicolumn{4}{c}{(V/nm)} & (eV/nm) & & & & \\"], rows[:-1])
print("TB tables written")
