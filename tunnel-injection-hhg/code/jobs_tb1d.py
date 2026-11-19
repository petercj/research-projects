"""Production runs of the 1D Au/ZnO tight-binding junction. Results: ../data_tb/<tag>.npz"""
import sys, os, numpy as np
from multiprocessing import Pool
from tb import *
p = dict(ZNO); Ac = calibrate_Ac(p)
DS = [1.5, 2.2, 3.0, 3.7, 4.6, 5.5, 6.4, 7.4, 8.3, 9.2, 11.0, 13.0]
jobs = []
for D in DS:
    jobs.append((f"A_c_D{D}", dict(D0=D, contact=True))); jobs.append((f"A_i_D{D}", dict(D0=D, contact=False)))
for ph in (0.2, 0.3, 0.5, 1.0, 1.3, 1.65, 2.0, 2.3, 2.6, 3.0):
    jobs.append((f"C_c_phi{ph}", dict(D0=5.5, contact=True, phiB=ph / EV)))
for ti in (0.05, 0.1, 0.2, 0.4, 0.75, 3.0):
    jobs.append((f"D_c_t{ti}", dict(D0=5.5, contact=True, tint=ti / EV)))
for D in (0.5, 0.82, 1.18, 1.56, 1.93):
    jobs.append((f"B_c_D{D}", dict(D0=D, contact=True, hartree=False))); jobs.append((f"B_i_D{D}", dict(D0=D, contact=False, hartree=False)))
for tag, kw in (("E_dt01", dict(dt=0.1)), ("E_long", dict(nc_layers=260, nm_layers=60)), ("E_metal", dict(nm_layers=140)),
                ("F_l4", dict(ell_nm=4.0)), ("F_l20", dict(ell_nm=20.0)), ("W_3200", dict(lam_nm=3200.0)), ("I_8cyc", dict(ncyc=8, nc_layers=260))):
    jobs.append((tag + "_c", dict(D0=5.5, contact=True, **kw))); jobs.append((tag + "_i", dict(D0=5.5, contact=False, **kw)))
# low-field references for subtracting the linear response (one per configuration)
DLIN = 0.05
lin = {}
for tag, kw in jobs:
    key = tag.rsplit("_D", 1)[0] if tag[0] in "AB" else tag
    kw2 = dict(kw); kw2["D0"] = DLIN
    lin[key + "_lin"] = kw2
jobs += list(lin.items())
jobs += [("G_c_D5.5", dict(D0=5.5, contact=True, snap=32)), ("G_i_D5.5", dict(D0=5.5, contact=False, snap=32))]
for D in (14.8, 18.5, 21.5, 25.0, 29.0):
    jobs.append((f"A_c_D{D}", dict(D0=D, contact=True))); jobs.append((f"A_i_D{D}", dict(D0=D, contact=False)))
# field scan with the depth profile of the near field below the edge of a 570 nm x 100 nm bar (nearfield.py)
def _profile():
    from compare_zno import near_field
    s_, Fz, Fabs, Exmax2, zp, ep = near_field()
    return (zp, ep)
PROF = _profile()
for D in (1.3, 1.9, 2.6, 3.2, 4.0, 4.8, 5.6, 6.5, 8.0, 10.0, 13.0, 17.0, 21.0, 25.0, 30.0):
    jobs.append((f"P_c_D{D}", dict(D0=D, contact=True, profile=PROF))); jobs.append((f"P_i_D{D}", dict(D0=D, contact=False, profile=PROF)))
jobs.append(("P_c_lin", dict(D0=DLIN, contact=True, profile=PROF))); jobs.append(("P_i_lin", dict(D0=DLIN, contact=False, profile=PROF)))

def job(a):
    tag, kw = a
    if not os.path.exists(os.path.join(DATA, "1d_" + tag + ".npz")):
        run_1d(Ac=Ac, tag="1d_" + tag, **kw)
    return tag
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    with Pool(n) as pool:
        for t in pool.imap_unordered(job, jobs):
            pass
    print(len(jobs), "runs done; A_c =", Ac / NM**2, "nm^2")
