"""Carrier densities left in the crystal after the pulse (chain model), from the stored occupations."""
import json
import numpy as np
from tb import build_1d
from figures_tb import L, f1, W1, scan, AU2CM2

_cache = {}
def weights(o, depth=0):
    nm = int((o["z"] < 0).sum()); nc = (len(o["z"]) - nm) // 2
    key = (bool(o["contact"]), nm, nc, str(o["params"]), float(o["Ac"]), depth)
    if key not in _cache:
        pp = json.loads(str(o["params"]))
        S, K = build_1d(pp, nm, nc, bool(o["contact"]), float(o["Ac"]))
        E, U = np.linalg.eigh(S.H0)
        deep = S.z >= depth * pp["b"] - 1e-9
        _cache[key] = (E, (U[deep] ** 2).sum(0))
    return _cache[key]

def carriers(o, depth=2):
    """(holes, electrons) per cm^2: depletion of initially occupied / population of initially empty
    eigenstates, weighted by their probability in the crystal beyond the first `depth` layers."""
    E, wc = weights(o, depth); nk = o["nk"]; assert np.allclose(E, o["E"], atol=1e-7)
    occ = E < 0; k = 2 / float(o["Ac"]) * AU2CM2
    return ((1 - nk[occ]) * wc[occ]).sum() * k, (nk[~occ] * wc[~occ]).sum() * k

if __name__ == "__main__":
    print(" D0   F0  | holes_c   elec_c   W_c    | holes_i   elec_i   W_i   (carriers in cm^-2, W in eV/nm^2)")
    rows = []
    for tag in scan("1d_A_c_D", "_D"):
        c, i = L(tag), L(tag.replace("_A_c_", "_A_i_")); hc, ec = carriers(c); hi, ei = carriers(i); F0 = f1(i)[1]
        rows.append((F0, hc, ec, W1(c), hi, ei, W1(i)))
        print(f"{float(c['D0']):5.1f} {F0:5.2f} | {hc:.2e} {ec:.2e} {W1(c):.4f} | {hi:.2e} {ei:.2e} {W1(i):.5f}")
    r = np.array(rows)
    print("local slopes d ln X / d ln I (I ~ F0^2):")
    for a in range(len(r) - 1):
        d = np.log(r[a + 1] / r[a]); print(f"  F0 {r[a,0]:.2f}-{r[a+1,0]:.2f}: holes_c {d[1]/d[0]/2:5.2f} elec_c {d[2]/d[0]/2:5.2f} W_c {d[3]/d[0]/2:5.2f} | holes_i {d[4]/d[0]/2:5.2f} W_i {d[6]/d[0]/2:5.2f}")
