# Kitaev Chain Solver — Implementation Spec

Implement a single Python module that diagonalizes the Kitaev chain for 4
conditions: {BdG, Majorana} × {OBC, PBC}. All formulas below are fully
derived — implement as given, do not re-derive or change conventions.

## 1. Physical Hamiltonian (fixed convention)

```
H = Σ_{j=1}^{N} [-μ(c_j†c_j - 1/2)]
  + Σ_{bonds} [ -t(c_j†c_{j+1} + c_{j+1}†c_j) - Δ(c_j c_{j+1} + c_{j+1}†c_j†) ]
```

- `t = 1.0` fixed. `μ`, `Δ` are free real floats. `N` = number of sites.
- OBC: bonds are `j = 1..N-1` (open chain, no wraparound).
- PBC: bonds are `j = 1..N` with site `N+1 ≡ 1` (ring).
- Both t and Δ are taken real (standard convention; do not generalize to
  complex phases).

This Hamiltonian is used to derive both representations below. Both
representations MUST produce identical eigenspectra for the same
`(N, μ, Δ, t, boundary)` — this is the primary correctness test (see §5).

## 2. Representation A: BdG (Nambu) operator basis

Nambu spinor: `Ψ = (c_1,...,c_N, c_1†,...,c_N†)^T` (length 2N).

```
H = (1/2) Ψ† H_BdG Ψ        (exact, no additive constant needed)
H_BdG = [[A,  B],
         [-B, -A]]          (2N x 2N, real, symmetric ⇒ Hermitian)
```

`A` is `N×N` real symmetric, `B` is `N×N` real antisymmetric.

Matrix elements (1-indexed, all unlisted entries are zero, fill both
`(i,j)` and `(j,i)` per symmetry/antisymmetry):

```
A[j,j]     = -μ                for j = 1..N
A[j,j+1]   = A[j+1,j] = -t     for each bond j (OBC: j=1..N-1; PBC adds j=N→1)

B[j,j+1]   = +Δ                for each bond j
B[j+1,j]   = -Δ                (antisymmetric partner)
```

For the PBC wraparound bond, `j=N, j+1 := 1`, i.e. set
`A[N,1]=A[1,N]=-t`, `B[N,1]=+Δ`, `B[1,N]=-Δ`.

### Diagonalization
`H_BdG` is real symmetric → use `scipy.linalg.eigh` (or `eigvalsh` if
eigenvectors not needed). Eigenvalues come in exact ± pairs `±E_n`,
`n=1..N`, `E_n ≥ 0`. Ground state energy = `-1/2 Σ E_n`.

## 3. Representation B: Majorana basis

Majorana operators per site: `a_j = c_j + c_j†`, `b_j = -i(c_j - c_j†)`,
Hermitian, `{γ_μ, γ_ν} = 2δ_{μν}`.

Substituting into H (already derived, use directly):

```
H = (i/4) γ^T A_M γ
```

where `γ = (a_1,b_1,a_2,b_2,...,a_N,b_N)` (length 2N, interleaved per
site), and `A_M` is `2N×2N` real antisymmetric. Using 1-indexed positions
`pos(a_j) = 2j-1`, `pos(b_j) = 2j`:

```
A_M[2j-1, 2j]   = -μ                        for j = 1..N   (on-site a_j–b_j)
A_M[2j,   2j-1] = +μ                        (antisymmetric partner)

A_M[2j-1, 2(j+1)]   = -(t+Δ)                for each bond j   (a_j – b_{j+1})
A_M[2(j+1), 2j-1]   = +(t+Δ)

A_M[2j,   2(j+1)-1] = -(t-Δ)                for each bond j   (b_j – a_{j+1})
A_M[2(j+1)-1, 2j]   = +(t-Δ)
```

For the PBC wraparound bond `j=N`, `j+1 := 1`: use positions
`pos(a_N)=2N-1`, `pos(b_N)=2N`, `pos(a_1)=1`, `pos(b_1)=2`, i.e.
`A_M[2N-1, 2] = -(t+Δ)`, `A_M[2N, 1] = -(t-Δ)`, plus antisymmetric partners.

### Diagonalization
`A_M` is real antisymmetric ⇒ `i·A_M` is Hermitian with eigenvalues
`±E_n` (real). Build `M = 1j * A_M` (complex128) and call
`scipy.linalg.eigvalsh(M)`. Eigenvalues must numerically match
Representation A's `H_BdG` eigenvalues exactly (same `±E_n` set).

## 4. Required API

Single module, e.g. `kitaev.py`:

```python
def build_hamiltonian(N: int, mu: float, delta: float, t: float,
                       representation: str,  # 'bdg' | 'majorana'
                       boundary: str          # 'obc' | 'pbc'
                       ) -> np.ndarray:
    """Returns the 2N x 2N matrix to diagonalize (H_BdG or i*A_M)."""

def spectrum(N: int, mu: float, delta: float, t: float = 1.0,
             representation: str = 'bdg', boundary: str = 'obc'
             ) -> dict:
    """
    Returns:
      'full'  : all 2N eigenvalues, sorted ascending (float64 ndarray)
      'E'     : the N non-negative quasiparticle energies, sorted ascending
      'E_gs'  : ground state energy = -0.5 * sum(E)
    """
```

`representation` and `boundary` are case-insensitive strings; raise
`ValueError` on anything else.

## 5. Performance requirements

- **No Python-level loops over site index for matrix construction.**
  Fill diagonals/off-diagonals with vectorized numpy ops
  (`np.arange`, fancy indexing, or `np.diag(v, k=±1)` + explicit
  corner-element assignment for the PBC wraparound term).
- Use `dtype=float64` for `H_BdG` construction; only cast to
  `complex128` for the Majorana matrix (`1j * A_M`).
- Diagonalize with `scipy.linalg.eigh`/`eigvalsh` (LAPACK), never a
  hand-rolled eigensolver.
- **Optional stretch optimization**: for OBC only, using the interleaved
  Majorana ordering, `A_M` is banded (bandwidth 3) — `scipy.linalg.eig_banded`
  gives O(N²) instead of O(N³). The BdG matrix can also be made banded
  under an interleaved `(c_1,c_1†,c_2,c_2†,...)` ordering (bandwidth ~4)
  for the same speedup. PBC breaks bandedness (corner coupling) — dense
  `eigh` only. Do not attempt this until dense correctness passes.

## 6. Correctness tests (must all pass)

1. **Cross-representation agreement**: for a grid of `(N, μ, Δ)` values
   (include `Δ=0`, `Δ=t`, `Δ≠t`, `μ=0`, `μ≠0`), BdG and Majorana full
   spectra match to `1e-10` for both OBC and PBC.
2. **Particle-hole symmetry**: full spectrum is symmetric about 0 to
   `1e-10`, for all 4 conditions.
3. **Hermiticity/structure checks**: `H_BdG` symmetric; `A_M`
   antisymmetric; both to machine precision after construction.
4. **Known topological limit** (OBC only): at `Δ=t`, as `N→large`
   (e.g. N=200), for `|μ| < 2t` the two smallest `|E_n|` are
   exponentially close to 0 (Majorana zero modes, topological phase);
   for `|μ| > 2t` the spectrum is gapped away from 0 (trivial phase).
5. **Gap closing** (PBC): the bulk gap closes at `μ = ±2t` (for generic
   `Δ≠0`) — smallest `E_n` → 0 there, for large N.

## 7. Deliverables

- `kitaev.py` — module per §4.
- `test_kitaev.py` — pytest covering §6.
- No plotting/CLI required unless requested separately.