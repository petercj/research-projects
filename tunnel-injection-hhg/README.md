# Sub-cycle rectification at the metal contact in antenna-enhanced solid-state HHG

Code, data and manuscript for a theory Letter on what a metal–semiconductor contact does to
high-harmonic generation (HHG) in the hot spot of a plasmonic nanoantenna.

## Contents

| Path | What it is |
|---|---|
| `paper/main.tex`, `paper/main.pdf` | The Letter (REVTeX 4.2, PRL format) |
| `paper/supplement.tex`, `paper/supplement.pdf` | Supplemental Material: model, numerics, convergence, extra runs |
| `paper/refs.bib` | Bibliography (entries checked against Crossref) |
| `code/junction.py` | The simulation: 1D real-space TDSE for a metal–crystal junction with the induced Hartree field |
| `code/jobs.txt` | Parameter sets of every run used in the paper |
| `code/figures.py` | Produces `figures/fig1.pdf` … `fig3.pdf` |
| `code/report_numbers.py` | Prints every number quoted in the text |
| `code/make_tables.py` | Writes the tables of the Supplemental Material |
| `data/*.npz` | Full output of every run (time traces, occupations, snapshots) |
| `data/summary.json`, `data/numbers.txt` | Scalar results of all runs and the numbers quoted in the text |
| `data/runlog.txt` | Console log of the production runs |

## Reproducing

```bash
pip install numpy scipy matplotlib
cd code
cat jobs.txt | xargs -P 4 -I{} sh -c "OMP_NUM_THREADS=1 python3 junction.py {}"
python3 figures.py
python3 report_numbers.py
python3 make_tables.py
python3 make_summary.py
cd ../paper && latexmk -pdf main.tex && latexmk -pdf supplement.tex
```

One run takes four to ten minutes on a single core; the full set of 49 runs is about four core-hours.
The `tag=` of each job is the file name of its output in `data/`.

## Model in one paragraph

Electrons move along the field line that leaves the antenna edge. The metal is a jellium slab, the
crystal a Mathieu-type periodic potential with a 3.34 eV gap, and the Fermi level sits a Schottky
barrier below the conduction band. The antenna's surface charge is an external sheet whose field
decays into the crystal over the near-field length. All occupied orbitals are propagated, and the
planar Hartree potential of the induced density is included self-consistently; it accounts for
dielectric screening, metallic screening and the space charge of injected carriers.

## Status

The manuscript has placeholders for the author list, affiliations and acknowledgments. APS asks
authors to disclose the use of AI tools in preparing a manuscript; this work was produced with
substantial AI assistance and the acknowledgment should say so.
