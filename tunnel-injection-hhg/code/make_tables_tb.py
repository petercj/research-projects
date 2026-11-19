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

def y1ref(pre, F0):
    for c in ("none", "right", "both"):
        if have(f"{pre}_F0.025_{c}"):
            l0 = L(f"{pre}_F0.025_{c}"); return yields(l0["t"], l0["Jx"], float(l0["w"]), [1])[0] * (abs(float(F0)) / 0.025) ** 2

def rows2d(pre, F0, confs, label):
    out = []
    for c, lab in confs:
        hi, lo = f"{pre}_F{F0}_{c}", f"{pre}_F0.025_{c}"
        if not (have(hi) and have(lo)): continue
        y1 = y1ref(pre, F0)
        o = L(hi); t, J, w = nl(hi, lo, "Jx"); Y = yields(t, J, w, range(1, 8))
        T = 2 * np.pi / w; n = int(o["ncyc"]); m = (o["t"] / T > n / 2 - 1) & (o["t"] / T < n / 2 + 1); F = o["F"]
        out.append(f"{label} & {F0} & {lab} & {np.abs(F[:, 0]).max():.2f} & {F[:, 3].max():.2f} & {-F[:, 3].min():.2f} & ${F[m, 3].mean():+.2f}$ & {float(o['Eabs'])*EV*2/float(o['Ly'])*NM:.3f} & "
                   + " & ".join(sci(Y[k - 1] / y1) for k in (2, 3, 4, 5)) + r" \\")
    return out

C3 = (("none", "insulated"), ("right", "one end"), ("both", "both ends"))
rows = []
for pre, F0, confs, label in (("G4", "0.75", C3, "33 columns"), ("G4b", "0.75", (("both", "both ends"),), "33 columns, wedge"), ("G4c8", "0.75", C3[1:], "33 columns, 8 cycles"),
                              ("G1", "0.5", C3, "32 columns"), ("G1", "0.75", C3, "32 columns"), ("G2", "0.75", C3[:2], "32 columns, 32 layers"), ("G1c8", "0.75", C3[1:], "32 columns, 8 cycles")):
    r = rows2d(pre, F0, confs, label)
    if r: rows += r + [r"\hline"]
write("tab_tb_2d.tex", [r"Strip & $F_{\rm m}$ & Contact & $|F_x|^{\rm out}$ & $F_{z,+}$ & $F_{z,-}$ & $F_{z,{\rm dc}}$ & $W$ & $S_2/S_1$ & $S_3/S_1$ & $S_4/S_1$ & $S_5/S_1$ \\",
                         r" & (V/nm) & & \multicolumn{4}{c}{(V/nm)} & (eV/nm) & & & & \\"], rows[:-1])

# mirror symmetry and the second harmonic
rows = []
for pre, label in (("G4", "5 cycles"), ("G4c8", "8 cycles")):
    if not (have(f"{pre}_F0.75_right") and have(f"{pre}_F-0.75_right") and have(f"{pre}_F0.025_right")): continue
    y1 = y1ref(pre, "0.75")
    t, Jp, w = nl(f"{pre}_F0.75_right", f"{pre}_F0.025_right", "Jx"); _, Jm, _ = nl(f"{pre}_F-0.75_right", f"{pre}_F0.025_right", "Jx")
    ent = [("one end in contact, $+F_{\\rm m}$", yields(t, Jp, w, range(1, 8)) / y1), ("one end in contact, $-F_{\\rm m}$", yields(t, Jm, w, range(1, 8)) / y1),
           ("two independent ends", yields(t, Jp - Jm, w, range(1, 8)) / (4 * y1))]
    if have(f"{pre}_F0.75_both") and have(f"{pre}_F0.025_both"):
        t2, Jb, w2 = nl(f"{pre}_F0.75_both", f"{pre}_F0.025_both", "Jx"); ent.append(("symmetric strip, both ends in contact", yields(t2, Jb, w2, range(1, 8)) / y1))
    if have(f"{pre}_F0.75_none") and have(f"{pre}_F0.025_none"):
        t3, Jn, w3 = nl(f"{pre}_F0.75_none", f"{pre}_F0.025_none", "Jx"); ent.append(("insulated", yields(t3, Jn, w3, range(1, 8)) / y1))
    for lab, Y in ent:
        rows.append(f"{label} & {lab} & " + " & ".join(sci(Y[k - 1]) for k in (2, 3, 4, 5)) + r" \\")
    rows.append(r"\hline")
if rows:
    write("tab_tb_sym.tex", [r"Pulse & Configuration & $S_2/S_1$ & $S_3/S_1$ & $S_4/S_1$ & $S_5/S_1$ \\"], rows[:-1])

# chain model with the computed near-field profile
rows = []
from carriers_tb import carriers
for tag in scan("1d_P_c_D", "_D"):
    ti = tag.replace("_P_c_", "_P_i_")
    if not have(ti): continue
    c, i = L(tag), L(ti); fm, fp, fd = f1(c); fmi, fpi, _ = f1(i); Yc, Yi = Y1(tag), Y1(ti)
    if fpi > 4.0: continue
    rows.append(f"{float(c['D0']):.1f} & {fpi:.2f} & {fmi:.2f} & {fm:.2f} & {fp:.2f} & {fd:.2f} & {W1(c):.4f} & {sci(W1(i))} & {sci(carriers(c)[0])} & {sci(carriers(i)[0])} & {Yc[1]/Yi[1]:.0f} \\\\")
write("tab_tb_prof.tex", [r"$D_0$ & $F_0$ & $F_-^{\rm ins}$ & $F_-$ & $F_+$ & $F_{\rm dc}$ & $W_{\rm contact}$ & $W_{\rm insulated}$ & \multicolumn{2}{c}{holes (cm$^{-2}$)} & $S_2$ ratio \\",
                           r"\multicolumn{6}{c}{(V/nm)} & \multicolumn{2}{c}{(eV/nm$^2$)} & contact & insulated & \\"], rows)
print("TB tables written")
