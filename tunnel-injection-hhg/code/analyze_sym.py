"""Second-harmonic cancellation for mirror-symmetric antennas (2D strip model).

A strip with one end in contact, driven by F(t), has the in-plane current J[F]. Its mirror image driven by
the same field has the current -J[-F]. Two such strips far apart form a mirror-symmetric antenna with
independent ends, whose current is J[F] - J[-F] (exact). This is compared with an explicitly
mirror-symmetric strip (both ends in contact, odd number of metal columns) and with single ends."""
import sys
import numpy as np
from tb import yields
from figures_tb import L, have, NM, EV

def nl(pre, F, c, flow="0.025"):
    """nonlinear part of the in-plane current for field amplitude F (string, may be negative)."""
    h, l = L(f"{pre}_F{F}_{c}"), L(f"{pre}_F{flow}_{c}")
    s = float(F) / float(flow)
    return h["t"], h["Jx"] - s * l["Jx"], float(h["w"]), s * l["Jx"]

def table(pre, F="0.75", orders=range(1, 10)):
    out = {}
    t, Jp, w, Jl = nl(pre, F, "right"); y1 = yields(t, Jl, w, [1])[0]
    out["one end, +F"] = yields(t, Jp, w, orders) / y1
    if have(f"{pre}_F-{F}_right"):
        _, Jm, _, _ = nl(pre, "-" + F, "right")
        out["one end, -F"] = yields(t, Jm, w, orders) / y1
        out["two independent ends (mirror pair)"] = yields(t, Jp - Jm, w, orders) / (4 * y1)
        out["  even part of one end"] = yields(t, 0.5 * (Jp + Jm), w, orders) / y1
    if have(f"{pre}_F{F}_both"):
        t2, Jb, w2, Jlb = nl(pre, F, "both"); out["both ends in contact (symmetric strip)"] = yields(t2, Jb, w2, orders) / yields(t2, Jlb, w2, [1])[0]
    if have(f"{pre}_F{F}_none") and have(f"{pre}_F0.025_none"):
        t3, Jn, w3, Jln = nl(pre, F, "none"); out["insulated"] = yields(t3, Jn, w3, orders) / yields(t3, Jln, w3, [1])[0]
    return out

if __name__ == "__main__":
    for pre in sys.argv[1:] or ["G4", "G4c8", "G4b"]:
        if not have(f"{pre}_F0.75_right") and not have(f"{pre}_F0.75_both"): continue
        print(pre)
        if have(f"{pre}_F0.75_right"):
            for k, Y in table(pre).items():
                print(f"  {k:42s} " + " ".join(f"S{n}={Y[n-1]:.1e}" for n in range(2, 8)))
        elif have(f"{pre}_F0.025_both"):
            t, J, w, Jl = nl(pre, "0.75", "both"); Y = yields(t, J, w, range(1, 10)) / yields(t, Jl, w, [1])[0]
            print(f"  {'both ends in contact':42s} " + " ".join(f"S{n}={Y[n-1]:.1e}" for n in range(2, 8)))
        for c in ("none", "right", "both"):
            for F in ("0.75", "-0.75"):
                tag = f"{pre}_F{F}_{c}"
                if not have(tag): continue
                o = L(tag); T = 2 * np.pi / float(o["w"]); Ly = float(o["Ly"]); Fz = o["F"]
                q = (o["QR"] + o["QL"]) * 2 / Ly * NM
                print(f"    {tag:22s} W={float(o['Eabs'])*EV*2/Ly*NM:.3f} eV/nm  Fz(R) +{Fz[:,3].max():.2f}/-{-Fz[:,3].min():.2f}  Fz(L) +{Fz[:,7].max():.2f}/-{-Fz[:,7].min():.2f}  q_end={q[-1]:.3f} e/nm (R {o['QR'][-1]*2/Ly*NM:.3f}, L {o['QL'][-1]*2/Ly*NM:.3f})")
