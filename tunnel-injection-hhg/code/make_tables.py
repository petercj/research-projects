"""Generate the LaTeX tables of the Supplemental Material from the simulation data."""
import os
import numpy as np
from common import *
from figures import fields, e2o, KA, DS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper")

def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"

def row(tag, label):
    d = load(tag); fm, fp, fdc = fields(d); h = harmonics(d)
    return (f"{label} & {fm:.2f} & {fp:.2f} & ${fdc:+.2f}$ & {d['Eabs']*EV*KA:.3f} & {e2o(d):.2f} & "
            f"{sci(h[3]/h[0])} & {sci(h[4]/h[0])} & {sci(h[8]/h[0])} \\\\")

def table(rows, fname):
    lines = [r"\begin{tabular}{lcccccccc}", r"\hline\hline",
             r"Run & $F_-$ & $F_+$ & $F_{\rm dc}$ & $W$ & even/odd & $S_4/S_1$ & $S_5/S_1$ & $S_9/S_1$ \\",
             r" & \multicolumn{3}{c}{(V/nm)} & (eV/nm$^2$) & & & & \\", r"\hline"]
    for tag, label in rows:
        if tag is None:
            lines.append(r"\hline")
        elif have(tag):
            lines.append(row(tag, label))
    lines += [r"\hline\hline", r"\end{tabular}"]
    open(os.path.join(OUT, fname), "w").write("\n".join(lines) + "\n")

table([("A_j_D5.7", "Reference"), ("E_j_dt005", r"$\Delta t=0.05$"), ("E_j_dx", r"$\Delta z=0.234$"),
       ("K_120_240", "Crystal 120 cells"), ("K_80_400", r"Metal 21 nm"), ("K_80_640", r"Metal 34 nm")], "tab_conv.tex")
table([("A_j_D5.7", r"Contact, $\ell=8$ nm"), ("A_s_D5.7", r"Insulated, $\ell=8$ nm"), ("E_b_D5.7", "No metal"), (None, None),
       ("F_j_l4.0", r"Contact, $\ell=4$ nm"), ("F_s_l4.0", r"Insulated, $\ell=4$ nm"),
       ("F_j_l16.0", r"Contact, $\ell=16$ nm"), ("F_s_l16.0", r"Insulated, $\ell=16$ nm"), (None, None),
       ("W_j_3200", r"Contact, 3.2 $\mu$m"), ("W_s_3200", r"Insulated, 3.2 $\mu$m"), (None, None),
       ("I_j_D5.7", "Contact, 8 cycles"), ("I_s_D5.7", "Insulated, 8 cycles")], "tab_extra.tex")

# field scan table
lines = [r"\begin{tabular}{ccccccccc}", r"\hline\hline",
         r"$D_0$ & $F_0$ & $F_-$ & $F_+$ & $F_{\rm dc}$ & $W_{\rm contact}$ & $W_{\rm insulated}$ & \multicolumn{2}{c}{even/odd} \\",
         r"\multicolumn{5}{c}{(V/nm)} & \multicolumn{2}{c}{(eV/nm$^2$)} & contact & insulated \\", r"\hline"]
for D in DS:
    if have(f"A_j_D{D}") and have(f"A_s_D{D}"):
        j, s = load(f"A_j_D{D}"), load(f"A_s_D{D}"); fm, fp, fdc = fields(j)
        lines.append(f"{D} & {epeak(s):.2f} & {fm:.2f} & {fp:.2f} & {fdc:.2f} & {j['Eabs']*EV*KA:.4f} & {s['Eabs']*EV*KA:.4f} & {e2o(j):.2f} & {e2o(s):.3f} \\\\")
lines += [r"\hline\hline", r"\end{tabular}"]
open(os.path.join(OUT, "tab_field.tex"), "w").write("\n".join(lines) + "\n")
print("tables written")
