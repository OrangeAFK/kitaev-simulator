# Kitaev Chain Solver

Diagonalizes the 1D Kitaev chain in BdG (Nambu) and Majorana representations, for open (OBC) and periodic (PBC) boundaries. Both representations return the same quasiparticle spectrum for given `(N, μ, Δ, t, boundary)`.

```python
from kitaev import spectrum

result = spectrum(N=20, mu=0.5, delta=0.8, t=1.0, representation="bdg", boundary="obc")
# result["full"]  — 2N eigenvalues, sorted ascending
# result["E"]     — N non-negative quasiparticle energies
# result["E_gs"]  — ground-state energy = -0.5 * sum(E)
```

Regenerate the figures and table numbers below with:

```bash
pip install -r requirements.txt
python plot_results.py
```

---

## Results

### Four variants at a glance

Same dense `eigvalsh` solver for all four cases; what changes is the matrix (BdG vs Majorana) and whether the wraparound bond is present (OBC vs PBC).

![Four variants](figures/four_variants.png)

**Read the 2×2 panel left-to-right / top-to-bottom:**

| | OBC | PBC |
|---|---|---|
| **BdG** | Edge Majorana zeros: `E_min ≈ 0` for `\|μ\| < 2` | No edge modes; gap closes only at `μ = ±2` |
| **Majorana** | Same physics as BdG + OBC (curves overlay) | Same physics as BdG + PBC (curves overlay) |

Snapshot at `N=80`, `Δ = t = 1` (`E_min`):

| Variant | μ=0 | μ=2 | μ=3 |
|---|---|---|---|
| BdG + OBC | ~10⁻¹⁵ (topological zero) | 0.039 | 1.004 |
| BdG + PBC | 2.0 (gapped) | ~10⁻¹⁵ (gap closing) | 1.0 |
| Majorana + OBC | ~0 (matches BdG+OBC) | 0.039 | 1.004 |
| Majorana + PBC | 2.0 (matches BdG+PBC) | ~10⁻¹⁵ | 1.0 |

BdG vs Majorana max spectral difference at `(μ=0.5, Δ=0.8, N=80)`: **8.9×10⁻¹⁵** (OBC), **5.8×10⁻¹⁵** (PBC).

So: **representation** is two bases for the same spectrum; **boundary** is the physically distinct choice (zeros vs bulk gap closing).

### Gap vs chemical potential (OBC)

At `Δ = t = 1`, the topological phase `|μ| < 2t` hosts Majorana zero modes: `E_min` is exponentially small. Outside that window the spectrum is gapped.

![Gap vs μ (OBC)](figures/gap_vs_mu_obc.png)

| Parameters | `E_min` |
|---|---|
| N=200, μ=0, Δ=1 (topological) | 4.4×10⁻¹⁶ |
| N=200, μ=3, Δ=1 (trivial) | 1.001 |

### Bulk gap closing (PBC)

Under periodic boundaries the bulk gap closes at the critical points `μ = ±2t` (here `Δ = 0.5`, `t = 1`).

![Gap closing (PBC)](figures/gap_closing_pbc.png)

| μ (N=100, Δ=0.5, PBC) | `E_min` |
|---|---|
| 0 | 1.0 |
| ±2 | ~10⁻¹⁶ |
| 3 | 1.0 |

### BdG ↔ Majorana agreement

Maximum absolute difference of the full spectra over a `(μ, Δ)` grid (OBC, N=20). Differences sit at machine precision (~10⁻¹⁴).

![BdG vs Majorana](figures/bdg_vs_majorana.png)

Max `|BdG − Majorana|` over the plotted grid: **2.3×10⁻¹⁴**.

### Sample spectrum

OBC, `N=10`, `μ=0`, `Δ=1`, `t=1`:

| | |
|---|---|
| `E` | `[0, 2, 2, 2, 2, 2, 2, 2, 2, 2]` |
| `E_gs` | −9.0 |
