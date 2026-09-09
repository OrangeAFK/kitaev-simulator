"""Correctness tests for the Kitaev chain solver."""

import numpy as np
import pytest
from scipy.linalg import eigvalsh

import kitaev

# ---------------------------------------------------------------------------
# Phase 0 — smoke / validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "representation,boundary,should_raise",
    [
        ("invalid", "obc", True),
        ("bdg", "invalid", True),
        ("BDG", "OBC", False),
    ],
    ids=["bad-repr", "bad-boundary", "case-insensitive-ok"],
)
def test_validation(representation, boundary, should_raise):
    if should_raise:
        with pytest.raises(ValueError):
            kitaev.build_hamiltonian(4, 0.0, 0.5, 1.0, representation, boundary)
    else:
        H = kitaev.build_hamiltonian(4, 0.0, 0.5, 1.0, representation, boundary)
        assert H.shape == (8, 8)


# ---------------------------------------------------------------------------
# Phase 1–2 — BdG structure
# ---------------------------------------------------------------------------

BDG_GRID = [
    (2, 0.0, 0.0),
    (5, 0.5, 0.5),
    (10, 0.0, 1.0),
    (20, 3.0, 0.5),
]

@pytest.mark.parametrize("N,mu,delta", BDG_GRID)
@pytest.mark.parametrize("boundary", ["obc", "pbc"])
def test_bdg_symmetric(N, mu, delta, boundary):
    H = kitaev.build_hamiltonian(N, mu, delta, 1.0, "bdg", boundary)
    np.testing.assert_allclose(H, H.T, atol=0.0)


@pytest.mark.parametrize("N,mu,delta", BDG_GRID)
@pytest.mark.parametrize("boundary", ["obc", "pbc"])
def test_particle_hole_symmetry_bdg(N, mu, delta, boundary):
    H = kitaev.build_hamiltonian(N, mu, delta, 1.0, "bdg", boundary)
    full = eigvalsh(H)
    np.testing.assert_allclose(full, -full[::-1], atol=1e-10)


# ---------------------------------------------------------------------------
# Phase 3–4 — Majorana structure + cross-representation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("N,mu,delta", BDG_GRID)
@pytest.mark.parametrize("boundary", ["obc", "pbc"])
def test_majorana_antisymmetric(N, mu, delta, boundary):
    A = kitaev._build_majorana_antisymmetric(N, mu, delta, 1.0, boundary)
    np.testing.assert_allclose(A, -A.T, atol=0.0)


CROSS_REPR_GRID = [
    (2, 0.0, 0.0),
    (2, 0.5, 0.5),
    (5, 0.0, 1.0),
    (5, 3.0, 0.5),
    (10, 0.5, 0.0),
    (20, 0.0, 0.5),
]

@pytest.mark.parametrize("N,mu,delta", CROSS_REPR_GRID)
@pytest.mark.parametrize("boundary", ["obc", "pbc"])
def test_cross_representation(N, mu, delta, boundary):
    spec_bdg = kitaev.spectrum(N, mu, delta, 1.0, "bdg", boundary)
    spec_maj = kitaev.spectrum(N, mu, delta, 1.0, "majorana", boundary)
    np.testing.assert_allclose(spec_bdg["full"], spec_maj["full"], atol=1e-10)


@pytest.mark.parametrize("representation", ["bdg", "majorana"])
@pytest.mark.parametrize("boundary", ["obc", "pbc"])
@pytest.mark.parametrize("N,mu,delta", [(5, 0.5, 0.5), (10, 0.0, 1.0)])
def test_particle_hole_symmetry(N, mu, delta, representation, boundary):
    result = kitaev.spectrum(N, mu, delta, 1.0, representation, boundary)
    full = result["full"]
    np.testing.assert_allclose(full, -full[::-1], atol=1e-10)


# ---------------------------------------------------------------------------
# Phase 5 — spectrum() API
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("representation", ["bdg", "majorana"])
@pytest.mark.parametrize("boundary", ["obc", "pbc"])
def test_spectrum_api(representation, boundary):
    N, mu, delta = 8, 0.3, 0.7
    result = kitaev.spectrum(N, mu, delta, 1.0, representation, boundary)
    assert set(result.keys()) == {"full", "E", "E_gs"}
    assert result["full"].shape == (2 * N,)
    assert result["E"].shape == (N,)
    assert np.all(result["E"] >= 0)
    np.testing.assert_allclose(result["E_gs"], -0.5 * result["E"].sum())


# ---------------------------------------------------------------------------
# Phase 6 — physics limits
# ---------------------------------------------------------------------------

def test_topological_zero_modes():
    t, delta = 1.0, 1.0
    N = 200

    topo = kitaev.spectrum(N, mu=0.0, delta=delta, t=t, boundary="obc")
    abs_eigs = np.sort(np.abs(topo["full"]))
    assert abs_eigs[0] < 1e-6
    assert abs_eigs[1] < 1e-6

    trivial = kitaev.spectrum(N, mu=3.0, delta=delta, t=t, boundary="obc")
    assert trivial["E"][0] > 0.1


def test_gap_closing_pbc():
    N, t, delta = 100, 1.0, 0.5

    at_gap = kitaev.spectrum(N, mu=2.0, delta=delta, t=t, boundary="pbc")
    assert at_gap["E"][0] < 1e-6

    gapped = kitaev.spectrum(N, mu=0.0, delta=delta, t=t, boundary="pbc")
    assert gapped["E"][0] > 0.01
