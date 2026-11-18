"""Write data/summary.json: scalar results and harmonic yields of every run."""
import os, json, glob
import numpy as np
from common import *
from figures import fields, e2o, KA
out = {}
for f in sorted(glob.glob(os.path.join(DATA, "*.npz"))):
    tag = os.path.basename(f)[:-4]; d = load(tag); fm, fp, fdc = fields(d); h = harmonics(d, range(1, 13))
    out[tag] = dict(params=d["p"], peak_internal_field_V_per_nm=epeak(d), F_minus=fm, F_plus=fp, F_dc=fdc,
                    absorbed_energy_eV_per_nm2=float(d["Eabs"]) * EV * KA, even_odd_ratio=float(e2o(d)),
                    harmonic_yields_1_to_12=[float(x) for x in h], norm_error=float(d["norm_err"]))
json.dump(out, open(os.path.join(DATA, "summary.json"), "w"), indent=1)
print(len(out), "runs summarised")
