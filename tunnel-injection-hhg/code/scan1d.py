import sys, numpy as np, time
from multiprocessing import Pool
from tb import *
p = dict(ZNO); Ac = calibrate_Ac(p); KA = 2 / Ac * NM**2
def job(a):
    D0, contact, kw = a
    o = run_1d(D0=D0, contact=contact, Ac=Ac, **kw)
    return D0, contact, fields_1d(o), o["Eabs"]*EV*KA, o["Y"]
if __name__ == "__main__":
    kw = eval(sys.argv[2]) if len(sys.argv) > 2 else {}
    Ds = [float(x) for x in sys.argv[1].split(",")]
    with Pool(4) as pool:
        res = pool.map(job, [(D, c, kw) for D in Ds for c in (True, False)])
    R = {(D, c): r for D, c, *r in res}
    print(" D0    F0 |  F-    F+    Fdc | F-/F-i F+/F+i Fdc/F0 | Eabs_c    Eabs_i   ratio | e/o_c e/o_i | c/i: H2 H3 H4 H5 H6 H7 H9")
    for D in Ds:
        (fm, fp, fdc), Ec, Yc = R[(D, True)]; (fmi, fpi, fdci), Ei, Yi = R[(D, False)]
        eo = lambda Y: (Y[3]+Y[5])/(Y[2]+Y[4])
        print(f"{D:5.1f} {fpi:5.2f} | {fm:5.2f} {fp:5.2f} {fdc:+5.2f} | {fm/fmi:5.2f} {fp/fpi:5.2f} {fdc/fpi:5.2f} | {Ec:8.4f} {Ei:8.5f} {Ec/Ei:6.1f} | {eo(Yc):5.2f} {eo(Yi):5.3f} | "
              + " ".join(f"{Yc[n-1]/Yi[n-1]:7.2g}" for n in (2,3,4,5,6,7,9)))
