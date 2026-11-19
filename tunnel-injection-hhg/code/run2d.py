"""usage: python run2d.py tag key=value ...  (keys: F0, contact, Nx, Nz, Wm, Hm, ncyc, dt, lam_nm, phiB_eV, cep, bevel)"""
import sys, time, numpy as np
from tb import *
tag = sys.argv[1]; kw = {}
for a in sys.argv[2:]:
    k, v = a.split("=")
    if k == "contact": kw[k] = v
    elif k == "phiB_eV": kw["phiB"] = float(v) / EV
    elif k in ("Nx", "Nz", "Wm", "Hm", "ncyc", "order", "snap", "bevel"): kw[k] = int(v)
    else: kw[k] = float(v)
p = dict(ZNO); kw.setdefault("Ly", 1.019 * p["b"])
o = run_2d(tag=tag, **kw)
i = np.argmax(np.abs(o["Fmacro"])); F = o["F"]
Yx = yields(o["t"], o["Jx"], o["w"]); YR = yields(o["t"], o["JxR"], o["w"]); YL = yields(o["t"], o["JxL"], o["w"])
Yz = yields(o["t"], np.gradient(o["Pz"], o["t"]), o["w"]); YzR = yields(o["t"], np.gradient(o["PzR"], o["t"]), o["w"])
print(f"[{tag}] {o['secs']:.0f}s norm={o['norm']:.1e} nocc={o['nocc']} norb={o['norb']} Eabs={o['Eabs']*EV:.4f} eV")
print(f"   peak macro {o['Fmacro'][i]:.3f}; local (Fx,Fz): Rout {F[i,0]:.2f},{F[i,1]:.2f} Rin {F[i,2]:.2f},{F[i,3]:.2f} Lout {F[i,4]:.2f},{F[i,5]:.2f} Lin {F[i,6]:.2f},{F[i,7]:.2f}; max|Fz Rin|={np.abs(F[:,3]).max():.2f}")
print(f"   charge in crystal right/left half: max {o['QR'].max():.4f}/{o['QL'].max():.4f} min {o['QR'].min():.4f}/{o['QL'].min():.4f} final {o['QR'][-1]:.4f}/{o['QL'][-1]:.4f}")
print("   Jx total  : " + " ".join(f"H{n}={Yx[n-1]/Yx[0]:.1e}" for n in range(2, 11)))
print("   Jx right  : " + " ".join(f"H{n}={YR[n-1]/Yx[0]:.1e}" for n in range(2, 11)))
print("   Jx left   : " + " ".join(f"H{n}={YL[n-1]/Yx[0]:.1e}" for n in range(2, 11)))
print("   Pz total  : " + " ".join(f"H{n}={Yz[n-1]/Yx[0]:.1e}" for n in range(2, 11)))
print("   Pz right  : " + " ".join(f"H{n}={YzR[n-1]/Yx[0]:.1e}" for n in range(2, 11)), flush=True)
