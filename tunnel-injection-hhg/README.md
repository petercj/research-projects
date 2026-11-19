# Sub-cycle rectification at the metal contact in antenna-enhanced solid-state HHG

Code, data and manuscript for a theory Letter on what a metal–semiconductor contact does to
high-harmonic generation (HHG) in the hot spot of a plasmonic nanoantenna.

## Contents

| Path | What it is |
|---|---|
| `paper/main.tex`, `paper/main.pdf` | The Letter (REVTeX 4.2, PRL format) |
| `paper/supplement.tex`, `paper/supplement.pdf` | Supplemental Material: models, numerics, tables, convergence, continuum-model check |
| `paper/refs.bib` | Bibliography (entries checked against Crossref) |
| `code/tb.py` | Main model: tight-binding gold–ZnO junction with the self-consistent Hartree field, as a chain (`run_1d`) and as a 2D strip array (`run_2d`) |
| `code/jobs_tb1d.py` | All chain-model runs of the paper (field, barrier and contact-hopping scans, convergence, low-field references) |
| `code/run2d.py` | Command-line driver for one 2D run |
| `code/figures_tb.py` | Produces `figures/fig1.pdf` … `fig3.pdf` and `figS_harm.pdf` |
| `code/make_tables_tb.py`, `code/report_tb.py` | Tables of the Supplemental Material and the numbers quoted in the text |
| `data_tb/*.npz` | Output of every tight-binding run (`1d_*`: chain; `G1_*`, `G2_*`, `G1c8_*`: 2D) |
| `code/junction.py`, `code/jobs.txt`, `code/figures.py`, `code/make_tables.py`, `code/report_numbers.py` | Independent continuum model (real-space grid) used as a check in the Supplement; writes `figures/figS1.pdf` … `figS3.pdf` |
| `data/*.npz`, `data/summary.json`, `data/numbers.txt` | Output of the continuum-model runs |

## Reproducing

```bash
pip install numpy scipy matplotlib
cd code
OMP_NUM_THREADS=1 python3 jobs_tb1d.py          # chain model, about 100 runs of 1-3 minutes each
for c in none right both; do for F in 0.025 0.5 0.75; do
  OMP_NUM_THREADS=1 python3 run2d.py G1_F${F}_${c} F0=$F contact=$c Nx=64 Nz=22 Wm=32 Hm=4
done; done                                      # 2D model, about 22 minutes per run
python3 figures_tb.py && python3 make_tables_tb.py
cat jobs.txt | xargs -P 4 -I{} sh -c "OMP_NUM_THREADS=1 python3 junction.py {}"   # continuum model
python3 figures.py && python3 make_tables.py
cd ../paper && latexmk -pdf main.tex && latexmk -pdf supplement.tex
```

The convergence runs of the 2D model use `Nz=32` (tags `G2_*`) and `ncyc=8` (tags `G1c8_*`).

## Models in one paragraph

Main model: a tight-binding crystal (cation s and anion p orbitals on a square lattice) with the
gap, effective masses, density and permittivity of ZnO, a single-band metal with the Fermi velocity
of gold, and a Schottky barrier of 0.7 eV. All occupied orbitals are propagated and the Hartree
potential of the induced charge is included self-consistently; it accounts for dielectric
screening, metallic screening and the space charge of injected carriers. In the chain geometry
the antenna near field is prescribed. In the 2D geometry only a uniform in-plane field is applied
and the near field at the edges of a metal strip results from the calculation. The continuum model
(jellium metal, Mathieu-type crystal potential, explicit insulating spacer) reproduces the results
independently.

## Status

The manuscript has placeholders for the author list, affiliations and acknowledgments. APS asks
authors to disclose the use of AI tools in preparing a manuscript; this work was produced with
substantial AI assistance and the acknowledgment should say so.
