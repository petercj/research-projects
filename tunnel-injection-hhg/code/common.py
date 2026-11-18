"""Shared helpers for analysis and figures."""
import os, json, glob
import numpy as np
from junction import spectrum, harm, EV, NM, VNM, DATA

AU2CM2 = 3.5711e16        # 1 a.u.^-2 in cm^-2

def load(tag):
    d = dict(np.load(os.path.join(DATA, tag + ".npz"), allow_pickle=False))
    d["p"] = json.loads(str(d["p"]))
    return d

def have(tag):
    return os.path.exists(os.path.join(DATA, tag + ".npz"))

def areal(n, d):
    """electrons per chain (per spin) -> areal density in cm^-2 (spin 2)."""
    return 2.0 * n / float(d["Ac"]) * AU2CM2

def spec(d, pad=4):
    return spectrum(d["t"], d["J"], float(d["w"]), pad)

def harmonics(d, orders=range(1, 16)):
    f, S = spec(d)
    return np.array([harm(f, S, n) for n in orders])

def epeak(d):
    """peak cell-averaged internal field (V/nm)."""
    return np.abs(d["Fin"]).max() * VNM
