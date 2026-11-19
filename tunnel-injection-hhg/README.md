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
| `code/jobs_2d_sym.txt`, `code/analyze_sym.py` | 2D runs and analysis for the mirror-symmetry test (symmetric strip, one end with both field signs, wedge, 8 cycles) |
| `code/nearfield.py` | Quasi-static near field of a metal bar on a dielectric (finite differences) |
| `code/compare_zno.py`, `code/carriers_tb.py` | Hot-spot average for the gold-on-ZnO antennas of Imasaka et al. and comparison with their measured intensity exponents; writes `figures/figS_zno.pdf` and `paper/tab_zno.tex` |
| `data_tb/*.npz` | Output of every tight-binding run (`1d_*`: chain; `G1_*`, `G2_*`, `G1c8_*`: 2D) |
| `code/junction.py`, `code/jobs.txt`, `code/figures.py`, `code/make_tables.py`, `code/report_numbers.py` | Independent continuum model (real-space grid) used as a check in the Supplement; writes `figures/figS1.pdf` … `figS3.pdf` |
| `data/*.npz`, `data/summary.json`, `data/numbers.txt` | Output of the continuum-model runs |

## Reproducing

```bash
pip install numpy scipy matplotlib
cd code
OMP_NUM_THREADS=1 python3 jobs_tb1d.py          # chain model, about 100 runs of 1-3 minutes each
cat jobs_2d_sym.txt | xargs -P 4 -I{} sh -c "OMP_NUM_THREADS=1 python3 run2d.py {}"   # 2D model, 22-35 minutes per run
python3 figures_tb.py && python3 make_tables_tb.py && python3 compare_zno.py
cat jobs.txt | xargs -P 4 -I{} sh -c "OMP_NUM_THREADS=1 python3 junction.py {}"   # continuum model
python3 figures.py && python3 make_tables.py
cd ../paper && latexmk -pdf main.tex && latexmk -pdf supplement.tex
```

The main 2D results use a strip of 33 metal columns (tags `G4*`), which is mirror symmetric down to
the lattice; run them with `cat jobs_2d_sym.txt | xargs -P 4 -I{} sh -c "python3 run2d.py {}"`.
Tags `G1_*`, `G2_*` (`Nz=32`) and `G1c8_*` (`ncyc=8`) are earlier runs with 32 columns, kept as a
variation of the geometry. `python3 compare_zno.py` reproduces the comparison with experiment
(the measured exponents in that script were read off a published figure, see the Supplement).

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
