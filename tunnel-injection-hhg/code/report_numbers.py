"""Print every number quoted in the manuscript and the Supplemental Material."""
import numpy as np
from common import *
from figures import fields, e2o, KA, DS, bulk_charge

EPS0 = 8.8541878e-12; QE = 1.602177e-19

def gauss(dF, eps):
    """sheet density (cm^-2) that produces a field step dF (V/nm) in a medium of permittivity eps."""
    return eps * EPS0 * dF * 1e9 / QE * 1e-4

def hot_fraction(d):
    """fraction of excess electrons above E_F with energy above E_c + E_g (junction eigenbasis)."""
    E = d["E"]; dn = d["nk"] - d["occ0"]; EF = float(d["EF"]); Ec = float(d["Ec"]); Eg = Ec - float(d["Ev"])
    up = E > EF
    return dn[up & (E > Ec + Eg)].sum() / dn[up].sum(), dn[up].sum()

print("=== field scan (contact vs insulated), 5-cycle pulse, 2.1 um, Phi_B = 0.7 eV")
print("  D0   F0   eps |  F-    F+   Fdc  | F-/F0 F+/F0 Fdc/F0 | Eabs_c  Eabs_i (eV/nm^2) ratio | e/o_c  e/o_i | ratios c/i: H3 H4 H5 H6 H7 H9 H11")
for D in DS:
    if not (have(f"A_j_D{D}") and have(f"A_s_D{D}")):
        continue
    j, s = load(f"A_j_D{D}"), load(f"A_s_D{D}")
    F0 = epeak(s); fm, fp, fdc = fields(j); hj, hs = harmonics(j), harmonics(s)
    print(f"{D:5.1f} {F0:5.2f} {D/F0:4.2f} | {fm:5.2f} {fp:5.2f} {fdc:5.2f} | {fm/F0:5.2f} {fp/F0:5.2f} {fdc/F0:5.2f} | "
          f"{j['Eabs']*EV*KA:7.4f} {s['Eabs']*EV*KA:7.4f} {j['Eabs']/s['Eabs']:6.1f} | {e2o(j):5.2f} {e2o(s):6.3f} | "
          + " ".join(f"{hj[n-1]/hs[n-1]:7.2g}" for n in (3, 4, 5, 6, 7, 9, 11)))

if have("A_j_D5.7"):
    j, s = load("A_j_D5.7"), load("A_s_D5.7")
    F0 = epeak(s); fm, fp, fdc = fields(j); eps = 5.7 / F0
    F = j["Fin"] * VNM; T = 2 * np.pi / float(j["w"]); t = j["t"] / T
    zc = [F[np.argmin(abs(t - x))] for x in (2.25, 2.75, 3.25, 3.75)]
    print(f"\n=== F0 = {F0:.2f} V/nm:  eps = {eps:.2f};  field at driver zero crossings (2.25,2.75,3.25,3.75 T): " + ", ".join(f"{x:.2f}" for x in zc), f"; after pulse {F[-1]:.2f}")
    print(f"    Gauss sheet density: clipped amplitude dF={F0-fm:.2f} -> {gauss(F0-fm, eps):.2e} cm^-2 ; dc field {fdc:.2f} -> {gauss(fdc, eps):.2e} cm^-2")
    print(f"    antenna surface charge eps0*D0/e = {gauss(5.7, 1.0):.2e} cm^-2")
    tj, qj = bulk_charge(j); ts_, qs_ = bulk_charge(s)
    dq = areal(qj, j) - np.interp(tj, ts_, areal(qs_, s))
    print(f"    excess (screened) bulk charge: max {dq.max():.2e}, min {dq.min():.2e}, mean over central 2 cycles {dq[(tj/T>1.5)&(tj/T<3.5)].mean():.2e} cm^-2")
    hf, ntot = hot_fraction(j)
    print(f"    excess electrons above E_F: {areal(ntot, j):.2e} cm^-2, fraction above E_c+E_g: {hf:.3f}")
    E = j["E"]; dn = j["nk"] - j["occ0"]; EF = float(j["EF"]); m = j["wm"] > 0.5
    print(f"    energy split: holes below E_F {((dn*(E-EF))[E<EF]).sum()/j['Eabs']:.2f}, electrons above {((dn*(E-EF))[E>EF]).sum()/j['Eabs']:.2f} of total")

print("\n=== no space-charge field (U_H = 0): contact")
for D in (0.5, 1.0, 1.5, 2.0):
    if have(f"B_j_D{D}"):
        b = load(f"B_j_D{D}"); fm, fp, fdc = fields(b)
        print(f"  D0={D}: F0={epeak(b):.2f}  Eabs={b['Eabs']*EV*KA:.4f} eV/nm^2  F-/F+={fm:.2f}/{fp:.2f}  excess e above E_F {areal(hot_fraction(b)[1], b):.2e} cm^-2  e/o={e2o(b):.2f}")

print("\n=== barrier scan at D0=5.7")
for p_ in (0.3, 0.5, 0.7, 1.0, 1.4, 1.67, 2.0, 2.3, 2.6):
    tag = "A_j_D5.7" if p_ == 0.7 else f"C_j_phi{p_}"
    if have(tag):
        d = load(tag); fm, fp, fdc = fields(d); h = harmonics(d)
        print(f"  phiB={p_:4.2f}: F-={fm:.2f} F+={fp:.2f} Fdc={fdc:+.3f}  Eabs={d['Eabs']*EV*KA:.4f}  e/o={e2o(d):.3f}  H4={h[3]:.1e} H6={h[5]:.1e} H9={h[8]:.1e}")

print("\n=== spacer thickness at D0=5.7 (relative to contact)")
if have("A_j_D5.7"):
    j = load("A_j_D5.7"); hj = harmonics(j)
    for dd in (0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5):
        tag = "A_s_D5.7" if dd == 1.5 else f"D_s_d{dd}"
        if have(tag):
            x = load(tag); h = harmonics(x); fm, fp, fdc = fields(x)
            print(f"  d={dd:5.3f} nm: Eabs/Eabs_c={x['Eabs']/j['Eabs']:.4f} (x{j['Eabs']/x['Eabs']:.1f} lower)  H4 ratio={h[3]/hj[3]:.4f}  H9 ratio={h[8]/hj[8]:.2f}  Fdc={fdc:+.3f}  F-/F+={fm:.2f}/{fp:.2f}  e/o={e2o(x):.3f}")

print("\n=== convergence / robustness at D0=5.7")
for tag, lab in (("A_j_D5.7", "reference (80 cells, Lm=240, dt=0.1, dx=0.3125)"), ("E_j_dt005", "dt=0.05"), ("E_j_dx", "dx=0.234"),
                 ("K_120_240", "120 cells"), ("K_80_400", "Lm=400"), ("K_80_640", "Lm=640"), ("E_b_D5.7", "bare crystal (no metal)"), ("A_s_D5.7", "insulated"),
                 ("F_j_l4.0", "contact ell=4"), ("F_s_l4.0", "insulated ell=4"), ("F_j_l16.0", "contact ell=16"), ("F_s_l16.0", "insulated ell=16"),
                 ("W_j_3200", "contact 3.2 um"), ("W_s_3200", "insulated 3.2 um"),
                 ("I_j_D5.7", "contact 8 cycles"), ("I_s_D5.7", "insulated 8 cycles"), ("I_j_D8.5", "contact 8 cycles D0=8.5"), ("I_s_D8.5", "insulated 8 cycles D0=8.5")):
    if have(tag):
        d = load(tag); fm, fp, fdc = fields(d); h = harmonics(d)
        print(f"  {lab:48s} F-={fm:.3f} F+={fp:.3f} Fdc={fdc:+.3f} Eabs={d['Eabs']*EV*KA:.4f} e/o={e2o(d):.3f} | " + " ".join(f"H{n}={h[n-1]:.2e}" for n in (3, 4, 5, 6, 7, 9)))

print("\n=== valence-band holes after the pulse (cm^-2): junction-eigenbasis count | bulk-window count (4 cells)")
for D in DS:
    if have(f"A_j_D{D}") and have(f"A_s_D{D}"):
        j, s = load(f"A_j_D{D}"), load(f"A_s_D{D}")
        print(f"  F0={epeak(s):.2f}: contact {areal(float(j['h_vb']), j):.2e} | {areal(j['Nh'][1], j):.2e}   insulated {areal(float(s['h_vb']), s):.2e} | {areal(s['Nh'][1], s):.2e}"
              f"   ratio {float(j['h_vb'])/max(float(s['h_vb']),1e-12):.1f} | {j['Nh'][1]/max(s['Nh'][1],1e-12):.1f}")
