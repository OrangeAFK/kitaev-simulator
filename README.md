# Numerical Kitaev Chain Study

Self-contained NumPy/SciPy/Matplotlib study of the noninteracting 1D Kitaev chain
in the quadratic \(2N\times 2N\) representation. Hopping is fixed at \(t=1\)
everywhere (not a configurable parameter). Transition points: \(\mu=\pm 2\).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Consistency checks (SPEC checklist); writes validation_report.txt
python scripts/validate_representations.py

# Regenerate all eight figures into figures/
python scripts/generate_figures.py
```

## Layout

```text
src/hamiltonians.py       # four Hamiltonians (fermionic/Majorana × OBC/PBC)
src/diagonalization.py    # spectra, gap, Bloch E(k)
src/wavefunctions.py      # site/Majorana weights, localization length
scripts/validate_representations.py
scripts/generate_figures.py
figures/
```

## Conventions

- Fermions: \(c_j^\dagger=(\gamma_{2j}+i\gamma_{2j+1})/2\)
- Majorana form: \(H=(i/4)\gamma^T A\gamma\) with \(A\) real antisymmetric
- Positive quasiparticle energies from eigenvalues of \(iA\) (or BdG)
- Near-zero tolerance: `ZERO_TOL = 1e-9` in `src/diagonalization.py`
- Localization length: least-squares fit \(\ln w_j = c - 2j/\xi\) on the left-edge envelope (undefined in the trivial phase)

## Figures

### Figure 1 — PBC bulk band structure

Majorana PBC Bloch bands at \(\mu=0\), \(\Delta=1\), overlaid with the analytic dispersion \(E_\pm(k)=\pm\sqrt{(\mu-2\cos k)^2+4\Delta^2\sin^2 k}\).

![Figure 1](figures/fig01_pbc_band_structure.png)

### Figure 2 — PBC vs OBC spectrum across \(\mu\)

Positive quasiparticle energies vs \(\mu\) for \(N=100\), \(\Delta=1\). OBC shows near-zero edge modes for \(|\mu|<2\); PBC does not.

![Figure 2](figures/fig02_pbc_vs_obc_spectrum.png)

### Figure 3 — Bulk gap and localization length

PBC gap closes at \(\mu=\pm 2\). OBC localization length \(\xi(\mu)\) grows toward the transition (fit documented above); undefined for \(|\mu|>2\).

![Figure 3](figures/fig03_gap_localization.png)

### Figure 4 — Representation / eigenstate comparison

Four panels at \(N=100\), \(\mu=0\), \(\Delta=1\): fermionic OBC, fermionic PBC, Majorana OBC, Majorana PBC. OBC states are edge-localized; PBC states are extended. Sorted positive spectra agree to numerical precision.

![Figure 4](figures/fig04_representation_comparison.png)

### Figure 5 — Majorana-component structure of the OBC zero mode

Weight on Majorana indices \(a=2,\ldots,2N+1\) at \(\mu=0\), \(\Delta=1\), concentrated on the unpaired end operators \(\gamma_2\) and \(\gamma_{2N+1}\).

![Figure 5](figures/fig05_majorana_components.png)

### Figure 6 — OBC wavefunction through the transition

Site weights for \(\mu=0,1.5,1.9,2.0,2.1,2.5\): localized MZM → broadened edge mode → critical delocalization → loss of the topological edge mode (post-transition states are not labeled MZMs).

![Figure 6](figures/fig06_wavefunction_transition.png)

### Figure 7 — Finite-size Majorana splitting

(A) \(E_{\min}\) vs \(N\) at \(\mu=0\) (exact zeros) with a \(\mu=1\) exponential guide. (B) \(E_{\min}(\mu)\) for \(N=20,50,100\).

![Figure 7](figures/fig07_finite_size_splitting.png)

### Figure 8 — Pairing-strength dependence

(A) PBC positive spectrum vs \(\Delta\) at \(\mu=0\). (B) OBC localization length vs \(\Delta\); \(\Delta=0\) is the nonsuperconducting limit (no MZM).

![Figure 8](figures/fig08_delta_dependence.png)
