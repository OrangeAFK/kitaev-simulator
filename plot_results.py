"""Generate README figures and print numeric table values."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from kitaev import spectrum

FIG_DIR = Path(__file__).resolve().parent / "figures"
T = 1.0


def gap_vs_mu_obc(N: int = 100) -> None:
    mus = np.linspace(-3.0, 3.0, 121)
    e_min = np.array(
        [spectrum(N, mu, delta=1.0, t=T, boundary="obc")["E"][0] for mu in mus]
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.semilogy(mus, np.maximum(e_min, 1e-16), color="#1f4e79", lw=2)
    ax.axvline(-2 * T, color="#888", ls="--", lw=1)
    ax.axvline(2 * T, color="#888", ls="--", lw=1)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$E_{\min}$")
    ax.set_title(rf"OBC gap vs $\mu$ ($N={N}$, $\Delta=t=1$)")
    ax.set_xlim(-3, 3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gap_vs_mu_obc.png", dpi=150)
    plt.close(fig)


def gap_closing_pbc(N: int = 100) -> None:
    mus = np.linspace(-3.0, 3.0, 121)
    delta = 0.5
    e_min = np.array(
        [spectrum(N, mu, delta=delta, t=T, boundary="pbc")["E"][0] for mu in mus]
    )

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(mus, e_min, color="#8b1a1a", lw=2)
    ax.axvline(-2 * T, color="#888", ls="--", lw=1, label=r"$\mu=\pm 2t$")
    ax.axvline(2 * T, color="#888", ls="--", lw=1)
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$E_{\min}$")
    ax.set_title(rf"PBC gap closing ($N={N}$, $\Delta={delta}$, $t=1$)")
    ax.set_xlim(-3, 3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gap_closing_pbc.png", dpi=150)
    plt.close(fig)


def bdg_vs_majorana(N: int = 20) -> None:
    mus = np.linspace(-2.5, 2.5, 26)
    deltas = np.linspace(0.0, 1.5, 16)
    diff = np.zeros((len(deltas), len(mus)))

    for i, delta in enumerate(deltas):
        for j, mu in enumerate(mus):
            bdg = spectrum(N, mu, delta, t=T, representation="bdg", boundary="obc")
            maj = spectrum(N, mu, delta, t=T, representation="majorana", boundary="obc")
            diff[i, j] = np.max(np.abs(bdg["full"] - maj["full"]))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.pcolormesh(mus, deltas, np.log10(np.maximum(diff, 1e-18)), shading="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}\max|\mathrm{BdG}-\mathrm{Majorana}|$")
    ax.set_xlabel(r"$\mu$")
    ax.set_ylabel(r"$\Delta$")
    ax.set_title(rf"BdG vs Majorana spectral difference (OBC, $N={N}$, $t=1$)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "bdg_vs_majorana.png", dpi=150)
    plt.close(fig)

    print(f"BdG–Majorana max |diff| over grid: {diff.max():.3e}")


def print_tables() -> None:
    print("\n=== Sample OBC spectrum (N=10, mu=0, Delta=1) ===")
    s = spectrum(10, mu=0.0, delta=1.0, t=T, boundary="obc")
    print("E =", np.array2string(s["E"], precision=6, suppress_small=True))
    print(f"E_gs = {s['E_gs']:.6f}")

    print("\n=== Topological vs trivial (OBC, Delta=t=1, N=200) ===")
    for mu in (0.0, 3.0):
        e0 = spectrum(200, mu=mu, delta=1.0, t=T, boundary="obc")["E"][0]
        print(f"mu={mu:g}: E_min = {e0:.6e}")

    print("\n=== PBC gap-closing snapshot (N=100, Delta=0.5, t=1) ===")
    for mu in (0.0, 2.0, -2.0, 3.0):
        e0 = spectrum(100, mu=mu, delta=0.5, t=T, boundary="pbc")["E"][0]
        print(f"mu={mu:g}: E_min = {e0:.6e}")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    gap_vs_mu_obc()
    gap_closing_pbc()
    bdg_vs_majorana()
    print_tables()
    print(f"\nWrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
