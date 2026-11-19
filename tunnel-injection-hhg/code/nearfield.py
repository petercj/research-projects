"""Quasi-static near field of a metal bar on a dielectric half-space (2D cross-section along the bar axis).

The bar (perfect conductor, length L, height h) lies on a substrate of permittivity eps_s and is exposed to
a uniform field E0 along its axis x. By symmetry the bar is at potential 0. The Laplace equation
div(eps grad phi) = 0 is solved by finite differences on a graded grid for x > 0 (phi = 0 at x = 0).
z points into the substrate; the contact edge is at (x, z) = (L/2, 0). Lengths in nm, fields in units of E0."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

def graded(points, hmin, hmax, lo, hi, ratio=1.12):
    """grid on [lo, hi] with spacing hmin near the given points, growing geometrically to hmax."""
    xs = set()
    for p in points:
        for sgn in (1, -1):
            x, h = p, hmin
            while lo - 1e-9 <= x <= hi + 1e-9:
                xs.add(round(x, 6)); x += sgn * h; h = min(h * ratio, hmax)
    xs |= {lo, hi}
    x = np.array(sorted(xs))
    keep = np.concatenate(([True], np.diff(x) > 0.45 * hmin))
    x = x[keep]; x[-1] = hi
    return x

def solve(L=500.0, h=50.0, eps_s=3.7, hmin=0.25, box=4000.0):
    x = graded([L / 2], hmin, 40.0, 0.0, box)
    z = graded([0.0, -h], hmin, 40.0, -box, box)
    nx, nz = len(x), len(z); X, Z = np.meshgrid(x, z, indexing="ij")
    metal = (X <= L / 2 + 1e-9) & (Z <= 1e-9) & (Z >= -h - 1e-9)
    # permittivity on cell faces: nodes at z > 0 are substrate, z <= 0 vacuum; face eps = harmonic mean
    epsn = np.where(Z > 1e-9, eps_s, 1.0)
    idx = np.arange(nx * nz).reshape(nx, nz)
    rows, cols, vals = [], [], []; rhs = np.zeros(nx * nz)
    hx = np.diff(x); hz = np.diff(z)
    def add(i, j, v): rows.append(i); cols.append(j); vals.append(v)
    dxc = np.empty(nx); dxc[1:-1] = 0.5 * (hx[1:] + hx[:-1]); dxc[0] = hx[0]; dxc[-1] = hx[-1]
    dzc = np.empty(nz); dzc[1:-1] = 0.5 * (hz[1:] + hz[:-1]); dzc[0] = hz[0]; dzc[-1] = hz[-1]
    fixed = metal.copy(); fixed[0, :] = True; fixed[-1, :] = True
    val = np.zeros((nx, nz)); val[-1, :] = -x[-1]                         # phi = -E0 x far away
    for i in range(nx):
        for j in range(nz):
            k = idx[i, j]
            if fixed[i, j]:
                add(k, k, 1.0); rhs[k] = val[i, j]; continue
            d = 0.0
            for (ii, jj, dist, area) in ((i + 1, j, hx[i], dzc[j]), (i - 1, j, hx[i - 1], dzc[j]),
                                         (i, j + 1, hz[j] if j + 1 < nz else None, dxc[i]), (i, j - 1, hz[j - 1] if j > 0 else None, dxc[i])):
                if jj < 0 or jj >= nz: continue                             # Neumann at top and bottom
                e = 2 * epsn[i, j] * epsn[ii, jj] / (epsn[i, j] + epsn[ii, jj])
                c = e * area / dist; add(k, idx[ii, jj], -c); d += c
            add(k, k, d)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(nx * nz, nx * nz))
    phi = spl.spsolve(A, rhs).reshape(nx, nz)
    return x, z, phi

def fields(x, z, phi):
    Ex = -np.gradient(phi, x, axis=0); Ez = -np.gradient(phi, z, axis=1)
    return Ex, Ez

if __name__ == "__main__":
    import sys
    L, h = (float(a) for a in sys.argv[1:3]) if len(sys.argv) > 2 else (500.0, 50.0)
    for hmin in (0.5, 0.25):
        x, z, phi = solve(L, h, hmin=hmin)
        Ex, Ez = fields(x, z, phi)
        jz = [np.argmin(abs(z - d)) for d in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)]
        print(f"L={L} h={h} hmin={hmin}: grid {len(x)} x {len(z)}")
        for s in (-20.0, -10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0):
            i = np.argmin(abs(x - (L / 2 + s)))
            print(f"   x - L/2 = {s:6.1f} nm: |E|/E0 at depth 0.5,1,2,5,10,20,50 nm: " + " ".join(f"{np.hypot(Ex[i, j], Ez[i, j]):6.2f}" for j in jz)
                  + "   Ez: " + " ".join(f"{Ez[i, j]:6.2f}" for j in jz[:4]))
