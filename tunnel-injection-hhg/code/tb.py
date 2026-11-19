"""Tight-binding model of a metal antenna on a ZnO-like crystal with the self-consistent Hartree field.

Crystal: rocksalt-type s-p model. Cations carry an s orbital, anions p orbitals; nearest-neighbour
s-p sigma hopping t_sp, second-neighbour p-p hopping t_pp. The gap is direct at Gamma,
E_g = eps_s - eps_p - 4 t_pp, the conduction mass is m_e = E_g / (8 t_sp^2 b^2), and the heavy-hole mass
is 1 / (4 t_pp b^2). Parameters are fitted to ZnO (E_g = 3.3 eV, m_e = 0.28, bond density of ZnO).
Metal: single-band model with Au-like Fermi velocity. Energies are measured from the Fermi level,
E_c = Phi_B. All quantities in atomic units.

Two geometries share the propagation code:
  build_1d : chain normal to the contact (the k_parallel = 0 slice of the 2D model) in a prescribed,
             decaying near field (external sheet at the metal surface), planar Hartree field.
  build_2d : periodic array of metal strips on a crystal slab. A uniform in-plane field enters as a
             Peierls phase; the near field at the strip edges is generated self-consistently by the
             Hartree potential (periodic 2D Poisson kernel for line charges).
"""
import os, sys, json, time
import numpy as np
import scipy.sparse as sp

EV = 27.211386
NM = 18.897261
VNM = 514.22067
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_tb")

ZNO = dict(
    b=0.228 * NM,        # bond length (site spacing); 2 b^3 = volume per ZnO formula unit
    Eg=3.3 / EV,         # direct gap
    me=0.28,             # conduction-band mass
    mhh=0.8,             # heavy-hole mass -> t_pp
    phiB=0.7 / EV,       # Schottky barrier E_c - E_F
    eps_inf=3.7,         # high-frequency permittivity (fixes the Coulomb scale)
    tm=2.0 / EV,         # metal hopping: v_F = 2 t_m b = 0.64 a.u. at mid band (Au: 0.64)
    EFm=4.0 / EV,        # metal Fermi level above the band bottom
    tint=1.5 / EV,       # metal-crystal hopping across the contact
)

def derived(p):
    b = p["b"]
    tsp = np.sqrt(p["Eg"] / (8 * p["me"] * b * b))
    tpp = 1.0 / (4 * p["mhh"] * b * b)
    eps_s = p["phiB"]
    eps_p = p["phiB"] - p["Eg"] - 4 * tpp
    return tsp, tpp, eps_s, eps_p

# ----------------------------------------------------------------------------- generic machinery
class System:
    """orbitals with positions, static Hamiltonian split into Peierls classes, Hartree kernel."""
    def __init__(self, n):
        self.n = n
        self.rows, self.cols, self.vals, self.dxs = [], [], [], []
        self.onsite = np.zeros(n)

    def hop(self, i, j, t, dx=0.0):
        """H_ij = t (and H_ji = t), dx = x_i - x_j (minimum image) for the Peierls phase."""
        self.rows += [i, j]; self.cols += [j, i]; self.vals += [t, t]; self.dxs += [dx, -dx]

    def finalize(self):
        rows, cols, vals, dxs = map(np.array, (self.rows, self.cols, self.vals, self.dxs))
        self.classes = []
        for dx in np.unique(np.round(dxs, 9)):
            m = np.isclose(dxs, dx)
            M = sp.csr_matrix((vals[m], (rows[m], cols[m])), shape=(self.n, self.n))
            self.classes.append((float(dx), M))
        self.H0 = sum(M for _, M in self.classes).toarray() + np.diag(self.onsite)

    def Hpsi(self, psi, A, V):
        out = V[:, None] * psi
        for dx, M in self.classes:
            out += (np.exp(1j * A * dx) if dx != 0 else 1.0) * (M @ psi)
        return out

    def Tpsi(self, psi, A):
        out = np.zeros_like(psi)
        for dx, M in self.classes:
            out += (np.exp(1j * A * dx) if dx != 0 else 1.0) * (M @ psi)
        return out

    def jx(self, psi, A, mask=None):
        """in-plane current  -<dH/dA> (per spin), optionally restricted to bonds with weight mask[i]*mask[j]."""
        tot = 0.0
        for dx, M in self.classes:
            if dx == 0:
                continue
            if mask is None:
                tot += dx * np.imag(np.exp(1j * A * dx) * np.vdot(psi, M @ psi))
            else:
                tot += dx * np.imag(np.exp(1j * A * dx) * np.vdot(psi * mask[:, None], M @ (psi * mask[:, None])))
        return tot        # = -<dH/dA>

def pulse(lam_nm=2000.0, ncyc=5, cep=0.0):
    w = 45.563353 / lam_nm
    T = ncyc * 2 * np.pi / w
    env = lambda t: np.sin(np.pi * t / T) ** 2 if 0 <= t <= T else 0.0
    f = lambda t: env(t) * np.cos(w * (t - T / 2) + cep)
    return w, T, f

def propagate(S, psi0, q_of, K, dt, T, A_of_t, Vext_of_t, hartree=True, observe=None, every=4, order=8):
    """Split-operator propagation of psi (n_orb x n_occ):  exp(-i V dt/2) exp(-i T dt) exp(-i V dt/2).

    V = on-site energies + external potential + Hartree potential of n(t) - n(0) is diagonal and is
    applied exactly; since it does not change the density, the Hartree potential evaluated after
    each hopping step is time centred. The hopping part T[A(t)] is applied with a dense propagator
    if it is time independent (S.UT) and by a Taylor series of the given order otherwise."""
    psi = psi0.astype(complex)
    q0 = q_of(psi)
    nsteps = int(round(T / dt)); dt = T / nsteps
    Vof = lambda t, dq: S.onsite + Vext_of_t(t) + (S.site_of @ (K @ dq) if hartree else 0.0)
    def kinetic(p, A):
        if getattr(S, "UT", None) is not None:
            return S.UT @ p
        out = p.copy(); term = p
        for n in range(1, order + 1):
            term = (-1j * dt / n) * S.Tpsi(term, A)
            out += term
        return out
    V = Vof(0.0, np.zeros(len(q0)))
    psi *= np.exp(-0.5j * dt * V)[:, None]
    rec = []
    for n in range(nsteps):
        psi = kinetic(psi, A_of_t((n + 0.5) * dt))
        t = (n + 1) * dt
        dq = q_of(psi) - q0
        V = Vof(t, dq)
        if observe is not None and n % every == 0:
            U = K @ dq if hartree else np.zeros(len(dq))
            rec.append(observe(t, psi * np.exp(-0.5j * dt * V)[:, None], dq, U))
        psi *= np.exp(-1j * dt * V * (1.0 if n < nsteps - 1 else 0.5))[:, None]
    return psi, q0, rec

def spectrum(t, acc, w, pad=4):
    a = acc * np.hanning(len(t)); n = pad * len(t)
    S = np.abs(np.fft.rfft(a, n) * (t[1] - t[0])) ** 2
    f = np.fft.rfftfreq(n, t[1] - t[0]) * 2 * np.pi / w
    return f, S

def harm(f, S, n):
    return S[(f > n - 0.5) & (f < n + 0.5)].sum() * (f[1] - f[0])

def yields(t, J, w, orders=range(1, 14)):
    f, S = spectrum(t, np.gradient(J, t), w)
    return np.array([harm(f, S, n) for n in orders])

# ----------------------------------------------------------------------------- 1D junction
def build_1d(p, nm_layers=60, nc_layers=160, contact=True, Ac=None):
    """metal layers at z = -nm*b .. -b, crystal layers at z = 0 .. (nc-1) b, two orbitals (s, p_z) each."""
    b = p["b"]; tsp, tpp, es, ep = derived(p)
    n = nm_layers + 2 * nc_layers
    S = System(n)
    z = np.empty(n); kind = np.empty(n, dtype="U1")
    im = lambda l: l                                   # metal orbital index
    is_ = lambda l: nm_layers + 2 * l
    ip = lambda l: nm_layers + 2 * l + 1
    for l in range(nm_layers):
        z[im(l)] = -(nm_layers - l) * b; kind[im(l)] = "m"; S.onsite[im(l)] = 2 * p["tm"] - p["EFm"]
        if l > 0:
            S.hop(im(l - 1), im(l), -p["tm"])
    for l in range(nc_layers):
        z[is_(l)] = z[ip(l)] = l * b; kind[is_(l)] = "s"; kind[ip(l)] = "p"
        S.onsite[is_(l)] = es; S.onsite[ip(l)] = ep
        if l > 0:
            S.hop(is_(l - 1), ip(l), -tsp); S.hop(ip(l - 1), is_(l), +tsp)     # sigma bonds along z
            S.hop(ip(l - 1), ip(l), 2 * tpp)                                    # k_par = 0 sum of p-p hops
    if contact and p["tint"] != 0:
        S.hop(im(nm_layers - 1), is_(0), -p["tint"]); S.hop(im(nm_layers - 1), ip(0), -p["tint"])
    S.finalize()
    S.z = z; S.kind = kind
    # one site per orbital in 1D (charges of s and p in a layer sit at the same z)
    S.site_of = sp.identity(n, format="csr")
    Ac = Ac if Ac is not None else 2 * b * b
    S.Ac = Ac
    K = -2 * np.pi * np.abs(z[:, None] - z[None, :]) * (2.0 / Ac)
    return S, K

def ground_state(S):
    E, U = np.linalg.eigh(S.H0)
    occ = E < 0.0
    return E, U, U[:, occ]

def chi0(E, U, nocc):
    """static independent-particle density response chi0_ij = dn_i / dV_j (per spin)."""
    Uv, Uc = U[:, :nocc], U[:, nocc:]
    W = 1.0 / (E[:nocc, None] - E[None, nocc:])                    # (v, c)
    M = np.einsum("iv,ic->ivc", Uv, Uc)                             # (site, v, c)
    return 2.0 * np.einsum("ivc,vc,jvc->ij", M, W, M, optimize=True)

def epsilon_1d(p, Ac):
    """static dielectric constant of the crystal chain (RPA): external sheet field / field inside."""
    S, K = build_1d(p, nm_layers=0, nc_layers=80, contact=False, Ac=Ac)
    E, U, psi = ground_state(S)
    X = chi0(E, U, psi.shape[1])
    Vext = S.z.copy()
    Vtot = np.linalg.solve(np.eye(S.n) - K @ X, Vext)
    zl = S.z[::2]; Vl = Vtot[::2]
    mid = (zl > zl.max() * 0.3) & (zl < zl.max() * 0.7)
    return 1.0 / np.polyfit(zl[mid], Vl[mid], 1)[0]

def calibrate_Ac(p):
    lo, hi = p["b"] ** 2 * 0.5, p["b"] ** 2 * 60
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        if epsilon_1d(p, mid) > p["eps_inf"]:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)

def run_1d(D0=1.0, contact=True, hartree=True, ell_nm=8.0, lam_nm=2000.0, ncyc=5, dt=0.2, nm_layers=60,
           nc_layers=160, p=None, Ac=None, tag=None, snap=0, **over):
    p = dict(ZNO if p is None else p); p.update(over)
    Ac = Ac if Ac is not None else calibrate_Ac(p)
    S, K = build_1d(p, nm_layers, nc_layers, contact, Ac)
    E, U, psi0 = ground_state(S)
    w, T, f = pulse(lam_nm, ncyc)
    b = p["b"]; z = S.z; ell = ell_nm * NM
    zI = -0.5 * b
    g = np.where(z > zI, ell * (1 - np.exp(-(z - zI) / ell)), 0.0)
    Dau = D0 / VNM
    q_of = lambda ps: (np.abs(ps) ** 2).sum(1)
    cz = z > zI
    l1, l2 = nm_layers + 2 * 2, nm_layers + 2 * 9            # s orbitals of crystal layers 2 and 9
    snaps_t, snaps = [], []; nxt = [0.0]; dsn = (2 * np.pi / w) / snap if snap else np.inf
    def observe(t, ps, dq, Uh):
        V = Dau * f(t) * g + Uh
        if t >= nxt[0]:
            snaps_t.append(t); snaps.append(dq.copy()); nxt[0] += dsn
        return (t, (dq * z).sum(), (V[l2] - V[l1]) / (z[l2] - z[l1]), dq[cz].sum())
    nst = int(round(T / dt)); dts = T / nst
    Th = S.H0 - np.diag(S.onsite)
    et, Ut = np.linalg.eigh(Th)
    S.UT = (Ut * np.exp(-1j * et * dts)) @ Ut.T
    psi, q0, rec = propagate(S, psi0, q_of, K, dt, T, lambda t: 0.0, lambda t: Dau * f(t) * g, hartree, observe)
    rec = np.array(rec); t = rec[:, 0]
    nk = (np.abs(U.conj().T @ psi) ** 2).sum(1)
    out = dict(t=t, P=rec[:, 1], F=rec[:, 2] * VNM, Q=rec[:, 3], w=w, T=T, Ac=Ac, E=E, nk=nk,
               Eabs=float(((nk - (E < 0)) * E).sum()), dq=q_of(psi) - q0, z=z,
               norm=float(np.abs((np.abs(psi) ** 2).sum(0) - 1).max()), params=json.dumps({k: float(v) for k, v in p.items()}),
               D0=D0, contact=contact, hartree=hartree, ell_nm=ell_nm, lam_nm=lam_nm, ncyc=ncyc,
               snaps_t=np.array(snaps_t), snaps=np.array(snaps), nm_layers=nm_layers, b=b)
    out["Y"] = yields(t, np.gradient(out["P"], t), w)
    if tag:
        os.makedirs(DATA, exist_ok=True); np.savez_compressed(os.path.join(DATA, tag + ".npz"), **out)
    return out

def fields_1d(o):
    F = o["F"]; t = o["t"] / (2 * np.pi / o["w"]); n = o["ncyc"]
    m = (t > n / 2 - 1) & (t < n / 2 + 1)
    return -F.min(), F.max(), F[m].mean()

if __name__ == "__main__":
    p = dict(ZNO)
    tsp, tpp, es, ep = derived(p)
    print(f"t_sp={tsp*EV:.3f} eV  t_pp={tpp*EV:.3f} eV  eps_s={es*EV:.2f} eV  eps_p={ep*EV:.2f} eV  b={p['b']/NM:.3f} nm")
    Ac = calibrate_Ac(p)
    print(f"A_c = {Ac/NM**2:.4f} nm^2 ({Ac/(2*p['b']**2):.2f} x 2b^2)  eps = {epsilon_1d(p, Ac):.3f}; eps at physical 2b^2: {epsilon_1d(p, 2*p['b']**2):.2f}")
    KA = 2 / Ac * NM ** 2                       # per chain and spin -> per nm^2
    for D0 in [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["3.7"])]:
        for contact in (True, False):
            t0 = time.time(); o = run_1d(D0=D0, contact=contact, Ac=Ac)
            fm, fp, fdc = fields_1d(o); Y = o["Y"]
            print(f"D0={D0} contact={contact}: F-={fm:.2f} F+={fp:.2f} Fdc={fdc:+.2f} V/nm  Eabs={o['Eabs']*EV*KA:.4f} eV/nm^2 norm={o['norm']:.1e} ({time.time()-t0:.0f}s)")
            print("    " + " ".join(f"H{n}={Y[n-1]/Y[0]:.1e}" for n in range(2, 12)))

# ----------------------------------------------------------------------------- 2D strip array
def build_2d(p, Nx=56, Nz=22, Wm=24, Hm=6, contact="both", Ly=None):
    """Periodic (in x) array of metal strips, Wm x Hm sites, on a crystal slab of Nx x Nz sites.

    contact: "both" (intimate contact), "none" (hopping across the contact switched off) or
             "right" (only the right half of the strip is in contact)."""
    b = p["b"]; tsp, tpp, es, ep = derived(p)
    assert Nx % 2 == 0
    i0 = (Nx - Wm) // 2; i1 = i0 + Wm - 1
    orb = {}                                  # (kind, i, j) -> orbital index
    xs, zs, kinds, site = [], [], [], []
    sites = {}
    def add(kind, i, j):
        if (i, j) not in sites:
            sites[(i, j)] = len(sites)
        orb[(kind, i, j)] = len(xs); xs.append(i * b); zs.append(j * b); kinds.append(kind); site.append(sites[(i, j)])
    for j in range(-Hm, 0):
        for i in range(i0, i1 + 1):
            add("m", i, j)
    for j in range(Nz):
        for i in range(Nx):
            if (i + j) % 2 == 0:
                add("s", i, j)
            else:
                add("x", i, j); add("z", i, j)
    n = len(xs); S = System(n)
    S.x = np.array(xs); S.z = np.array(zs); S.kind = np.array(kinds); S.site = np.array(site); S.nsite = len(sites)
    S.sites = sites; S.orb = orb; S.geom = dict(Nx=Nx, Nz=Nz, Wm=Wm, Hm=Hm, i0=i0, i1=i1, b=b)
    for (kind, i, j), o in orb.items():
        if kind == "m":
            S.onsite[o] = 4 * p["tm"] - p["EFm"]
            if ("m", i + 1, j) in orb: S.hop(o, orb[("m", i + 1, j)], -p["tm"], -b)
            if ("m", i, j + 1) in orb: S.hop(o, orb[("m", i, j + 1)], -p["tm"], 0.0)
            if j == -1 and p["tint"] != 0:
                ok = contact == "both" or (contact == "right" and i > (i0 + i1) / 2)
                if ok:
                    tgt = ("s", i, 0) if ("s", i, 0) in orb else ("z", i, 0)
                    S.hop(o, orb[tgt], -p["tint"], 0.0)
        elif kind == "s":
            S.onsite[o] = es
            ip = (i + 1) % Nx; im = (i - 1) % Nx
            S.hop(o, orb[("x", ip, j)], -tsp, -b); S.hop(o, orb[("x", im, j)], +tsp, +b)
            if j + 1 < Nz: S.hop(o, orb[("z", i, j + 1)], -tsp, 0.0)
            if j - 1 >= 0: S.hop(o, orb[("z", i, j - 1)], +tsp, 0.0)
        else:
            S.onsite[o] = ep
            for di in (1, -1):                # p-p hops to the diagonal neighbours in the next row
                if j + 1 < Nz:
                    S.hop(o, orb[(kind, (i + di) % Nx, j + 1)], tpp, -di * b)
    S.finalize()
    S.site_of = sp.csr_matrix((np.ones(n), (np.arange(n), S.site)), shape=(n, S.nsite))
    # one complex CSR with a phase per non-zero for the Peierls substitution
    rows, cols, vals, dxs = map(np.array, (S.rows, S.cols, S.vals, S.dxs))
    M = sp.csr_matrix((np.arange(1, len(vals) + 1), (rows, cols)), shape=(n, n))
    order = M.data - 1
    S._T = sp.csr_matrix((vals[order].astype(complex), M.indices, M.indptr), shape=(n, n))
    S._vals = vals[order]; S._dx = dxs[order]
    def Tpsi(psi, A):
        S._T.data = S._vals * np.exp(1j * A * S._dx)
        return S._T @ psi
    S.Tpsi = Tpsi
    S._Tx = S._T.copy()
    def jloc(psi, A):
        """bond currents -dH/dA assigned to the row orbital (per spin); their sum is the total J_x."""
        S._Tx.data = S._vals * S._dx * np.exp(1j * A * S._dx)
        return np.imag((psi.conj() * (S._Tx @ psi)).sum(1))
    S.jloc = jloc
    # periodic 2D Coulomb kernel between line charges on the sites
    sx = np.empty(S.nsite); sz = np.empty(S.nsite)
    for (i, j), k in sites.items():
        sx[k] = i * b; sz[k] = j * b
    S.sx, S.sz = sx, sz
    Lam = Nx * b; Ly = Ly if Ly is not None else b; S.Ly = Ly; S.Lam = Lam
    dxm = sx[:, None] - sx[None, :]; dzm = sz[:, None] - sz[None, :]
    arg = 2 * (np.cosh(2 * np.pi * dzm / Lam) - np.cos(2 * np.pi * dxm / Lam))
    np.fill_diagonal(arg, (2 * np.pi * 0.447 * b / Lam) ** 2)
    K = -(2.0 / Ly) * np.log(arg)
    return S, K

def epsilon_2d(p, Ly, Nx=16, Nz=24):
    """RPA dielectric constant of the 2D crystal slab for a field along z."""
    S, K = build_2d(p, Nx=Nx, Nz=Nz, Wm=2, Hm=0, contact="none", Ly=Ly)
    E, U = np.linalg.eigh(S.H0); nocc = int((E < 0).sum())
    X = chi0(E, U, nocc)
    Xs = S.site_of.T @ (S.site_of.T @ X).T                 # aggregate orbitals -> sites
    Vext = S.sz.copy()
    Vtot = np.linalg.solve(np.eye(S.nsite) - K @ Xs, Vext)
    zl = np.unique(S.sz); Vl = np.array([Vtot[S.sz == z_].mean() for z_ in zl])
    mid = (zl > zl.max() * 0.3) & (zl < zl.max() * 0.7)
    return 1.0 / np.polyfit(zl[mid], Vl[mid], 1)[0]

def calibrate_Ly(p):
    lo, hi = 0.2 * p["b"], 40 * p["b"]
    for _ in range(18):
        mid = np.sqrt(lo * hi)
        if epsilon_2d(p, mid) > p["eps_inf"]:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)

def run_2d(F0=0.4, contact="both", lam_nm=2000.0, ncyc=5, dt=0.4, order=6, Nx=56, Nz=22, Wm=24, Hm=6,
           Ly=None, p=None, tag=None, hartree=True, snap=8, **over):
    """F0: amplitude of the macroscopic in-plane field (V/nm)."""
    p = dict(ZNO if p is None else p); p.update(over)
    S, K = build_2d(p, Nx, Nz, Wm, Hm, contact, Ly)
    E, U = np.linalg.eigh(S.H0); occ = E < 0
    psi0 = U[:, occ]
    w, T, f = pulse(lam_nm, ncyc)
    tt = np.linspace(0, T, 20001); Fau = F0 / VNM
    At = np.concatenate(([0.0], np.cumsum(0.5 * (np.array([f(x) for x in tt[1:]]) + np.array([f(x) for x in tt[:-1]])) * np.diff(tt)))) * Fau
    A_of_t = lambda t: np.interp(t, tt, At)
    b = p["b"]; g = S.geom; i0, i1 = g["i0"], g["i1"]; xc = 0.5 * (i0 + i1) * b
    q_of = lambda ps: S.site_of.T @ (np.abs(ps) ** 2).sum(1)
    cryst = S.kind != "m"
    right_orb = (S.x > xc).astype(float); left_orb = 1.0 - right_orb
    right_site = S.sx > xc
    cs = S.sz >= 0
    # probes for the local field: crystal rows 0..3, four columns outside / under each strip edge
    idx = np.full((Nx, Nz), -1)
    for (i, j), k in S.sites.items():
        if j >= 0: idx[i, j] = k
    def local_field(U, Fm, cols):
        Ug = U[idx]                                          # (Nx, Nz) potential on crystal sites
        Fx = (np.roll(Ug, -1, 0) - np.roll(Ug, 1, 0)) / (2 * b) + Fm
        Fz = np.gradient(Ug, b, axis=1)
        sel = np.ix_(np.array(cols) % Nx, np.arange(0, 4))
        return Fx[sel].mean(), Fz[sel].mean()
    probes = dict(Rout=range(i1 + 1, i1 + 5), Rin=range(i1 - 3, i1 + 1), Lout=range(i0 - 4, i0), Lin=range(i0, i0 + 4))
    snaps_t, snaps = [], []
    nextsnap = [0.0]; dsnap = (2 * np.pi / w) / snap
    def observe(t, ps, dq, Uh):
        A = A_of_t(t); Fm = Fau * f(t)
        pop = (np.abs(ps) ** 2).sum(1)
        jl = S.jloc(ps, A)
        row = [t, jl.sum(), (jl * right_orb).sum(), (jl * left_orb).sum(),
               (dq * S.sz).sum(), (dq * S.sz)[right_site].sum(), (dq * S.sz)[~right_site].sum(),
               dq[cs & right_site].sum(), dq[cs & ~right_site].sum()]
        for k in ("Rout", "Rin", "Lout", "Lin"):
            row += list(local_field(Uh, Fm, probes[k]))
        if t >= nextsnap[0]:
            snaps_t.append(t); snaps.append(dq.copy()); nextsnap[0] += dsnap
        return row
    t0 = time.time()
    psi, q0, rec = propagate(S, psi0, q_of, K, dt, T, A_of_t, lambda t: 0.0, hartree, observe, every=4, order=order)
    rec = np.array(rec); t = rec[:, 0]
    nk = (np.abs(U.conj().T @ psi) ** 2).sum(1)
    out = dict(t=t, Jx=rec[:, 1], JxR=rec[:, 2], JxL=rec[:, 3], Pz=rec[:, 4], PzR=rec[:, 5], PzL=rec[:, 6],
               QR=rec[:, 7], QL=rec[:, 8], F=rec[:, 9:17] * VNM, w=w, T=T, Ly=S.Ly, Eabs=float(((nk - occ) * E).sum()),
               snaps_t=np.array(snaps_t), snaps=np.array(snaps), sx=S.sx, sz=S.sz, Fmacro=np.array([F0 * f(x) for x in t]),
               norm=float(np.abs((np.abs(psi) ** 2).sum(0) - 1).max()), secs=time.time() - t0, F0=F0, contact=contact,
               Nx=Nx, Nz=Nz, Wm=Wm, Hm=Hm, b=b, ncyc=ncyc, lam_nm=lam_nm, nocc=int(occ.sum()), norb=S.n)
    if tag:
        os.makedirs(DATA, exist_ok=True); np.savez_compressed(os.path.join(DATA, tag + ".npz"), **out)
    return out
