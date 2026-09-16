"""Wavefunction weights and localization-length extraction."""

from __future__ import annotations

import numpy as np

from .diagonalization import (
    ZERO_TOL,
    diagonalize_bdg,
    diagonalize_majorana,
    extract_low_energy_states,
    e_min_obc,
)
from .hamiltonians import (
    build_fermionic_obc,
    build_fermionic_pbc,
    build_majorana_obc,
    build_majorana_pbc,
)


def site_weights_majorana(psi: np.ndarray) -> np.ndarray:
    """Physical-site weights from a Majorana eigenvector: |psi_{2j}|^2 + |psi_{2j+1}|^2."""
    psi = np.asarray(psi).ravel()
    n2 = len(psi)
    assert n2 % 2 == 0
    N = n2 // 2
    w = np.zeros(N, dtype=float)
    for j in range(N):
        w[j] = abs(psi[2 * j]) ** 2 + abs(psi[2 * j + 1]) ** 2
    return w


def site_weights_bdg(evec: np.ndarray) -> np.ndarray:
    """Physical-site weights from a BdG eigenvector (u, v): |u_j|^2 + |v_j|^2."""
    evec = np.asarray(evec).ravel()
    n2 = len(evec)
    assert n2 % 2 == 0
    N = n2 // 2
    u = evec[:N]
    v = evec[N:]
    return np.abs(u) ** 2 + np.abs(v) ** 2


def majorana_component_weights(psi: np.ndarray) -> np.ndarray:
    """|psi_a|^2 for Majorana array indices (labels a = index+2)."""
    return np.abs(np.asarray(psi).ravel()) ** 2


def lowest_subspace_weights_majorana(
    N: int, mu: float, delta: float, pbc: bool
) -> tuple[np.ndarray, np.ndarray, float]:
    """Gauge-invariant site weights of the lowest-|E| Majorana subspace.

    Returns (site_weights, majorana_weights, E_ref) where weights are summed
    |psi|^2 over the near-degenerate low-energy eigenvectors and renormalized.
    """
    A = build_majorana_pbc(N, mu, delta) if pbc else build_majorana_obc(N, mu, delta)
    evals, evecs = diagonalize_majorana(A)
    sub_e, sub_v = extract_low_energy_states(evals, evecs)
    maj_w = np.zeros(2 * N, dtype=float)
    site_w = np.zeros(N, dtype=float)
    for i in range(sub_v.shape[1]):
        maj_w += majorana_component_weights(sub_v[:, i])
        site_w += site_weights_majorana(sub_v[:, i])
    # Renormalize to sum 1 for plotting clarity
    if maj_w.sum() > 0:
        maj_w /= maj_w.sum()
    if site_w.sum() > 0:
        site_w /= site_w.sum()
    e_ref = float(np.min(np.abs(sub_e)))
    return site_w, maj_w, e_ref


def lowest_subspace_weights_bdg(
    N: int, mu: float, delta: float, pbc: bool
) -> tuple[np.ndarray, float]:
    """Gauge-invariant site weights of the lowest-|E| BdG subspace."""
    H = build_fermionic_pbc(N, mu, delta) if pbc else build_fermionic_obc(N, mu, delta)
    evals, evecs = diagonalize_bdg(H)
    sub_e, sub_v = extract_low_energy_states(evals, evecs)
    site_w = np.zeros(N, dtype=float)
    for i in range(sub_v.shape[1]):
        site_w += site_weights_bdg(sub_v[:, i])
    if site_w.sum() > 0:
        site_w /= site_w.sum()
    e_ref = float(np.min(np.abs(sub_e)))
    return site_w, e_ref


def edge_concentration(site_w: np.ndarray, edge_sites: int = 5) -> float:
    """Fraction of weight on the first/last edge_sites sites."""
    N = len(site_w)
    e = min(edge_sites, N // 2)
    return float(site_w[:e].sum() + site_w[-e:].sum())


def compute_localization_length(
    N: int, mu: float, delta: float
) -> float | None:
    """Fit localization length xi from OBC near-zero edge envelope.

    Fits ln w_j = c - 2*j/xi over the left-half sites with appreciable weight.
    Returns None in the trivial phase or when no edge mode is present.
    """
    if abs(delta) < ZERO_TOL:
        return None
    if abs(mu) >= 2.0 - 1e-12:
        return None

    emin = e_min_obc(N, mu, delta)
    site_w, _, _ = lowest_subspace_weights_majorana(N, mu, delta, pbc=False)

    # Require a low-energy OBC mode; do not fabricate xi in the trivial phase.
    if emin > 0.2:
        return None
    # Prefer edge-ish states, but allow broad critical modes (|mu| near 2).
    if edge_concentration(site_w) < 0.15 and abs(mu) < 1.5:
        return None

    half = N // 2
    w = site_w[:half].copy()
    if w.max() <= 0:
        return None

    # Exactly solvable / strongly localized: weight only on the first site(s).
    if w[0] > 0.99 * w.sum():
        return 0.5

    thresh = 1e-12 * w.max()
    js = []
    lns = []
    for i in range(half):
        if w[i] > thresh:
            j = i + 1  # physical site index
            js.append(j)
            lns.append(np.log(w[i]))
    if len(js) < 2:
        return None

    js_arr = np.asarray(js, dtype=float)
    lns_arr = np.asarray(lns, dtype=float)
    # ln w = a + b * j  with b = -2/xi
    b, _a = np.polyfit(js_arr, lns_arr, 1)
    if b >= -1e-14:
        return None
    xi = -2.0 / b
    if not np.isfinite(xi) or xi <= 0:
        return None
    return float(xi)
