"""Numbers from the tight-binding (Au/ZnO) runs: 1D scans and 2D strip array."""
import os, glob, numpy as np
from tb import *
L = lambda tag: dict(np.load(os.path.join(DATA, tag + ".npz"), allow_pickle=False))
have = lambda tag: os.path.exists(os.path.join(DATA, tag + ".npz"))
eo = lambda Y: (Y[3] + Y[5]) / (Y[2] + Y[4])
def f1(o):
    F = o["F"]; t = o["t"] / (2 * np.pi / float(o["w"])); n = int(o["ncyc"]); m = (t > n / 2 - 1) & (t < n / 2 + 1)
    return -F.min(), F.max(), F[m].mean()
def KA(o): return 2 / float(o["Ac"]) * NM ** 2 * EV
if __name__ == "__main__":
    print("=== 1D field scan (contact | insulated)")
    print("  D0    F0 |  F-    F+    Fdc | F-/F-i F+/F0 Fdc/F0 |  W_c     W_i (eV/nm^2) ratio | e/o_c e/o_i | c/i: H2 H3 H4 H5 H6 H7 H9")
    for f in sorted(glob.glob(os.path.join(DATA, "1d_A_c_D*.npz")), key=lambda s: float(s.split("_D")[1][:-4])):
        D = f.split("_D")[1][:-4]
        if not have(f"1d_A_i_D{D}"): continue
        c, i = L(f"1d_A_c_D{D}"), L(f"1d_A_i_D{D}"); fm, fp, fd = f1(c); fmi, fpi, _ = f1(i); Yc, Yi = c["Y"], i["Y"]
        print(f"{float(D):5.1f} {fpi:5.2f} | {fm:5.2f} {fp:5.2f} {fd:+5.2f} | {fm/fmi:5.2f} {fp/fpi:5.2f} {fd/fpi:5.2f} | {float(c['Eabs'])*KA(c):7.4f} {float(i['Eabs'])*KA(i):8.5f} {float(c['Eabs'])/float(i['Eabs']):6.1f} | {eo(Yc):5.2f} {eo(Yi):5.3f} | "
              + " ".join(f"{Yc[n-1]/Yi[n-1]:7.2g}" for n in (2, 3, 4, 5, 6, 7, 9)))
    ref = L("1d_A_i_D5.5") if have("1d_A_i_D5.5") else None
    for title, pat, key in (("barrier scan (D0=5.5)", "1d_C_c_phi*", "_phi"), ("contact hopping scan (D0=5.5)", "1d_D_c_t*", "_t")):
        print("===", title)
        for f in sorted(glob.glob(os.path.join(DATA, pat + ".npz")), key=lambda s: float(s.split(key)[-1][:-4])):
            c = L(os.path.basename(f)[:-4]); fm, fp, fd = f1(c); Y = c["Y"]
            print(f"   {key[1:]}={f.split(key)[-1][:-4]:>5s}: F-={fm:.2f} F+={fp:.2f} Fdc={fd:+.3f} W={float(c['Eabs'])*KA(c):.4f} e/o={eo(Y):.2f} H4/H1={Y[3]/Y[0]:.1e} H6/H1={Y[5]/Y[0]:.1e}")
    print("=== other 1D runs (D0=5.5 unless noted)")
    for f in sorted(glob.glob(os.path.join(DATA, "1d_[BEFWI]_*.npz"))):
        tag = os.path.basename(f)[:-4]; c = L(tag); fm, fp, fd = f1(c); Y = c["Y"]
        print(f"   {tag:16s}: F-={fm:.3f} F+={fp:.3f} Fdc={fd:+.3f} W={float(c['Eabs'])*KA(c):.5f} e/o={eo(Y):.3f} " + " ".join(f"H{n}={Y[n-1]/Y[0]:.1e}" for n in (3, 4, 5, 6, 7, 9)))
