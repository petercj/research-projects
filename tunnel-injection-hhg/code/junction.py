"""Metal | semiconductor junction in an antenna near field: 1D real-space TDSE with induced Hartree field.

Coordinate z runs along the field line leaving the metal (z < 0: metal, z > 0: crystal). Atomic units.

  H(t) = p^2/2 + V_static(z) + D(t) g(z) + U_H[dn](z, t)

* V_static: jellium metal (depth set by the metal Fermi energy), Mathieu-type crystal potential
  -V0 [1 + cos(2 pi z / a)], optional insulating spacer, vacuum outside (electron affinity chi).
* D(t) g(z): field of the antenna surface charge, treated as an external sheet source at the metal
  surface; g(z) = ell (1 - exp(-z/ell)) for z > 0 and 0 inside the metal (near-field decay length ell).
* U_H: planar Hartree potential of the induced density dn = n(t) - n(0), U_H'' = -4 pi dn_3D with
  dn_3D = 2 dn_1D / A_c (spin 2, one chain per transverse area A_c = a^2).  Switch off with hartree=0.
* Occupied orbitals: every eigenstate of the static Hamiltonian below E_F = E_c - Phi_B.

Usage:  python junction.py key=value ...   (see DEFAULTS).  Results go to ../data/<tag>.npz
"""
import os, sys, time, json, hashlib
import numpy as np
from scipy.linalg import eigh
import scipy.fft as sfft

EV = 27.211386
NM = 18.897261
VNM = 514.22067           # 1 a.u. of field in V/nm
NW = int(os.environ.get("NW", "1"))
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

DEFAULTS = dict(
    variant="junction",   # junction | spacer | bare
    D0=1.0,               # peak external (sheet) field, V/nm
    lam=2100.0,           # wavelength, nm
    ncyc=5,               # sin^2 pulse, total duration in cycles
    cep=0.0,              # carrier-envelope phase (rad)
    phiB=0.7,             # Schottky barrier E_c - E_F, eV
    V0=0.32,              # crystal potential amplitude (a.u.)  -> gap 3.34 eV
    a=8.0,                # lattice constant (a.u.)
    ncell=80,             # crystal length in cells
    Lm=240.0,             # metal thickness (a.u.)
    EFm=5.5,              # metal Fermi energy above band bottom, eV
    chi=4.0,              # electron affinity of the crystal, eV
    dsp=0.0,              # spacer thickness, nm
    hsp=3.1,              # spacer barrier above E_c, eV
    gsp=0.2564,           # slope of g inside the spacer (1/eps_spacer for a passive oxide, eps=3.9)
    ell=8.0,              # near-field decay length, nm
    hartree=1,            # induced Hartree field on/off
    N=3072, Ltot=960.0,   # grid (dx = 0.3125 a.u.)
    dt=0.1,
    every=4,
    snap=4,               # density/potential snapshots per optical cycle
    tag="",
)

def band_edges(V0, a, npw=41, nk=201):
    """Bulk bands of -V0[1+cos(2 pi z/a)]: returns (bands[nk,4], k)."""
    G = 2 * np.pi / a * (np.arange(npw) - npw // 2)
    ks = np.linspace(-np.pi / a, np.pi / a, nk)
    bands = np.empty((nk, 5))
    off = -V0 / 2 * (np.eye(npw, k=1) + np.eye(npw, k=-1))
    for i, k in enumerate(ks):
        bands[i] = np.linalg.eigvalsh(np.diag(0.5 * (k + G) ** 2 - V0) + off)[:5]
    return bands, ks

def build(p):
    N, Ltot, a = p["N"], p["Ltot"], p["a"]
    dx = Ltot / N
    Ls = p["ncell"] * a
    assert Ltot >= p["Lm"] + 30.0 + Ls + 40.0 + (p["dsp"] * NM if p["variant"] == "spacer" else 0.0), "grid too short"
    dsp = p["dsp"] * NM if p["variant"] == "spacer" else 0.0
    left = p["Lm"] + 30.0
    z = np.arange(N) * dx - left
    bands, _ = band_edges(p["V0"], a)
    Ev, Ec = bands[:, 1].max(), bands[:, 2].min()
    EF = Ec - p["phiB"] / EV
    Vm = EF - p["EFm"] / EV
    Vout = Ec + p["chi"] / EV
    sm = lambda x, w=1.5: 0.5 * (1 + np.tanh(x / w))
    in_sc = sm(z - dsp) * sm(dsp + Ls - z)
    Vsc = -p["V0"] * (1 + np.cos(2 * np.pi * (z - dsp) / a))
    V = Vout + in_sc * (Vsc - Vout)
    if p["variant"] in ("junction", "spacer"):
        V += sm(z + p["Lm"]) * sm(-z) * (Vm - Vout)
    if p["variant"] == "spacer" and dsp > 0:
        V += sm(z) * sm(dsp - z) * (Ec + p["hsp"] / EV - Vout)
    return dict(z=z, dx=dx, V=V, Ev=Ev, Ec=Ec, EF=EF, Ls=Ls, dsp=dsp, bands=bands)

def eigenstates(s):
    z, dx, V = s["z"], s["dx"], s["V"]
    N = len(z)
    k = 2 * np.pi * sfft.fftfreq(N, dx)
    F = sfft.fft(np.eye(N), axis=0, norm="ortho")
    T = (F.conj().T * (0.5 * k ** 2)) @ F
    E, U = eigh(T.real + np.diag(V))
    return E, U, k

def tagof(p):
    if p["tag"]:
        return p["tag"]
    key = {k: v for k, v in p.items() if k != "tag" and v != DEFAULTS[k]}
    body = "_".join(f"{k}{v}" for k, v in sorted(key.items())) or "default"
    return body if len(body) < 120 else hashlib.md5(body.encode()).hexdigest()

def run(**kw):
    p = dict(DEFAULTS); p.update(kw)
    s = build(p)
    z, dx = s["z"], s["dx"]
    E, U, k = eigenstates(s)
    occ = E < s["EF"]
    psi = U[:, occ].T.astype(complex) / np.sqrt(dx)
    norb = psi.shape[0]
    wm = (U[z < 0] ** 2).sum(0)
    # reference basis: eigenstates of the same crystal with the metal removed (bound states only)
    pb = dict(p); pb["variant"] = "bare"
    sb = build(pb)
    if s["dsp"] > 0:                                   # same crystal position as in the spacer variant
        sb["V"] = np.interp(z - s["dsp"], z, sb["V"], left=sb["V"][0], right=sb["V"][-1])
    Eb, Ub, _ = eigenstates(sb)
    Vout = s["Ec"] + p["chi"] / EV
    loc = (Ub[z > s["dsp"] - 6.0] ** 2).sum(0) > 0.9
    selc = (Eb > s["EF"]) & (Eb < Vout) & loc
    selv = (Eb < s["EF"]) & loc
    Pc = Ub[:, selc] * np.sqrt(dx); Pv = Ub[:, selv] * np.sqrt(dx); Ecb = Eb[selc]
    # carriers are counted in the bulk of the crystal: orbitals are multiplied by a smooth window that
    # removes the first m cells next to the contact (metal-induced gap states, interface dipole)
    sm_ = lambda x, w_=0.5 * p["a"]: 0.5 * (1 + np.tanh(x / w_))
    MW = (2, 4, 8)
    wins = [sm_(z - s["dsp"] - m * p["a"]) for m in MW]
    def carriers(psi_):
        out_ = []
        for wv in wins:
            chi = psi_.conj() * wv
            out_.append(((np.abs(chi @ Pc) ** 2).sum(0), (np.abs(chi @ Pv) ** 2).sum(0)))
        return out_
    car0 = carriers(psi)
    w = 45.563353 / p["lam"]
    Tp = p["ncyc"] * 2 * np.pi / w
    nsteps = int(round(Tp / p["dt"])); dt = Tp / nsteps
    ell = p["ell"] * NM
    dsp = s["dsp"]
    g = np.where(z > dsp, p["gsp"] * dsp + ell * (1 - np.exp(-np.clip(z - dsp, 0, None) / ell)),
                 np.where(z > 0, p["gsp"] * z, 0.0))
    D0 = p["D0"] / VNM
    field = lambda t: D0 * np.sin(np.pi * t / Tp) ** 2 * np.cos(w * (t - Tp / 2) + p["cep"])
    expT = np.exp(-1j * 0.5 * k ** 2 * dt)
    Vs = s["V"]
    z0 = s["dsp"]
    right = z > z0
    Ac = p["a"] ** 2
    rho0 = (np.abs(psi) ** 2).sum(0)
    q0 = rho0[right].sum() * dx
    # window for the cell-averaged internal field: cells 1..4 of the crystal
    i1 = np.searchsorted(z, z0 + 1 * p["a"]); i2 = np.searchsorted(z, z0 + 5 * p["a"])
    every = p["every"]
    nrec = (nsteps + every - 1) // every
    ts = np.empty(nrec); J = np.empty(nrec); Q = np.empty(nrec); Fin = np.empty(nrec); Dt = np.empty(nrec)
    cevery = max(1, int(round(4.0 / dt)) // every * every) or every   # carrier trace every ~4 a.u.
    tc, Nc_t, Nh_t = [], [], []
    j1 = np.searchsorted(z, z0 + 8 * p["a"]); j2 = np.searchsorted(z, z0 + 12 * p["a"])
    Fdeep = np.empty(nrec)
    quarter = (2 * np.pi / w) / p["snap"]; nextsnap = quarter; snaps_t, snaps_dn, snaps_U = [], [], []
    UH = np.zeros_like(z)
    psi *= np.exp(-0.5j * dt * (Vs + field(0.0) * g))
    t0 = time.time(); r = 0
    for n in range(nsteps):
        pk = sfft.fft(psi, axis=1, workers=NW)
        rec = (n % every == 0)
        if rec:
            J[r] = ((np.abs(pk) ** 2) @ k).sum() * dx / len(z)
            ts[r] = n * dt + dt / 2
        pk *= expT
        psi = sfft.ifft(pk, axis=1, workers=NW)
        tn = (n + 1) * dt
        rho = (np.abs(psi) ** 2).sum(0)
        if p["hartree"]:
            dn3 = 2.0 * (rho - rho0) / Ac
            Ef = -4 * np.pi * np.cumsum(dn3) * dx                 # dU/dz at half-integer points
            UH[1:] = np.cumsum(Ef[:-1]) * dx                    # symmetric (energy-conserving) kernel
        Vdyn = field(tn) * g + UH
        psi *= np.exp(-1j * dt * (Vs + Vdyn) * (1.0 if n < nsteps - 1 else 0.5))
        if rec:
            Q[r] = rho[right].sum() * dx - q0
            Fin[r] = (Vdyn[i2] - Vdyn[i1]) / (z[i2] - z[i1])       # cell-averaged total field (a.u.)
            Dt[r] = field(tn)
            Fdeep[r] = (Vdyn[j2] - Vdyn[j1]) / (z[j2] - z[j1])
            r += 1
        if n % cevery == 0 and n < nsteps - 1:
            car = carriers(psi * np.exp(0.5j * dt * (Vs + Vdyn)))      # undo half of the potential step
            tc.append(tn)
            Nc_t.append([c[0].sum() - c0[0].sum() for c, c0 in zip(car, car0)])
            Nh_t.append([c0[1].sum() - c[1].sum() for c, c0 in zip(car, car0)])
        if tn >= nextsnap:
            snaps_t.append(tn); snaps_dn.append(rho - rho0); snaps_U.append(Vdyn.copy()); nextsnap += quarter
    C = (psi.conj() @ U) * np.sqrt(dx)
    nk_ = (np.abs(C) ** 2).sum(0)
    is_m = wm > 0.5
    Ev, Ec, EF = s["Ev"], s["Ec"], s["EF"]
    cb = (~is_m) & (E > EF); vb = (~is_m) & (E < EF)
    hot = cb & (E > Ec + (Ec - Ev))
    rhoT = (np.abs(psi) ** 2).sum(0)
    carT = carriers(psi)
    Eg = Ec - Ev
    Nc = np.array([c[0].sum() - c0[0].sum() for c, c0 in zip(carT, car0)])
    Nh = np.array([c0[1].sum() - c[1].sum() for c, c0 in zip(carT, car0)])
    dnc = carT[1][0] - car0[1][0]                         # per-state CB occupation, 4-cell window
    chi = psi * wins[1]
    rho_c = (np.abs((chi @ Pc.conj()) @ Pc.T) ** 2).sum(0) / dx ** 2 * dx   # density of CB-projected part
    EH = 0.5 * np.sum(UH * (rhoT - rho0)) * dx
    out = dict(p=json.dumps(p), t=ts[:r], J=J[:r], Q=Q[:r], Fin=Fin[:r], D=Dt[:r], w=w, Tp=Tp,
               E=E, nk=nk_, wm=wm, occ0=occ.astype(float), Ev=Ev, Ec=Ec, EF=EF, z=z, drho=rhoT - rho0,
               n_cb=nk_[cb].sum(), h_vb=(1 - nk_[vb]).sum(), h_m=(1 - nk_[is_m & (E < EF)]).sum(),
               e_m=nk_[is_m & (E > EF)].sum(), n_hot=nk_[hot].sum(), norb=norb,
               norm_err=abs(nk_.sum() - norb), secs=time.time() - t0, Ac=Ac,
               Nc=Nc, Nh=Nh, MW=np.array(MW), Ecb=Ecb, dnc=dnc, rho_c=rho_c, EH=EH,
               Nc_hot=dnc[Ecb > Ec + Eg].sum(), tc=np.array(tc), Nc_t=np.array(Nc_t), Nh_t=np.array(Nh_t),
               Fdeep=Fdeep[:r], snaps_t=np.array(snaps_t), snaps_dn=np.array(snaps_dn), snaps_U=np.array(snaps_U),
               Eabs=float(((nk_ - occ) * E).sum()))
    os.makedirs(DATA, exist_ok=True)
    np.savez_compressed(os.path.join(DATA, tagof(p) + ".npz"), **out)
    return out

def spectrum(t, J, w, pad=4):
    """Power spectrum of dJ/dt (Hann window); frequency axis in harmonic orders."""
    acc = np.gradient(J, t) * np.hanning(len(t))
    n = pad * len(t)
    S = np.abs(np.fft.rfft(acc, n) * (t[1] - t[0])) ** 2
    f = np.fft.rfftfreq(n, t[1] - t[0]) * 2 * np.pi / w
    return f, S

def harm(f, S, n):
    return S[(f > n - 0.5) & (f < n + 0.5)].sum() * (f[1] - f[0])

if __name__ == "__main__":
    kw = {}
    for arg in sys.argv[1:]:
        k_, v = arg.split("=", 1)
        kw[k_] = type(DEFAULTS[k_])(v) if not isinstance(DEFAULTS[k_], str) else v
    o = run(**kw)
    p = json.loads(o["p"])
    f, S = spectrum(o["t"], o["J"], o["w"])
    Fpk = np.abs(o["Fin"]).max() * VNM
    print(f"[{tagof(p)}] norb={o['norb']} {o['secs']:.0f}s norm_err={o['norm_err']:.1e}  peak internal field={Fpk:.3f} V/nm")
    print("   bulk carriers (window 2/4/8 cells): Nc=" + "/".join(f"{x:.5f}" for x in o["Nc"]) + "  Nh=" + "/".join(f"{x:.5f}" for x in o["Nh"])
          + f"  hot={o['Nc_hot']:.5f} | metal holes={o['h_m']:.5f}  Eabs={o['Eabs']*EV:.4f} eV  EH={o['EH']*EV:.4f} eV")
    print("   " + " ".join(f"H{n}={harm(f,S,n):.2e}" for n in range(1, 14)), flush=True)
