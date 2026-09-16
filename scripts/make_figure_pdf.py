#!/usr/bin/env python3
"""Combine the figures into a single printable PDF (one figure per page)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "figures"
OUT = FIGDIR / "kitaev_figures.pdf"

PAGES = [
    ("fig01_pbc_band_structure.png", "Figure 1 - PBC bulk band structure"),
    ("fig02_pbc_vs_obc_spectrum.png", "Figure 2 - PBC vs OBC spectrum across mu"),
    ("fig03_gap_localization.png", "Figure 3 - Bulk gap and localization length"),
    ("fig04_representation_comparison.png", "Figure 4 - Representation / eigenstate comparison"),
    ("fig05_wavefunction_transition.png", "Figure 5 - OBC wavefunction through the transition"),
    ("fig06_finite_size_splitting.png", "Figure 6 - Finite-size Majorana splitting"),
    ("fig07_delta_dependence.png", "Figure 7 - Pairing-strength dependence"),
]

PAGESIZE = (8.5, 11.0)  # US Letter portrait, inches


def main() -> None:
    missing = [name for name, _ in PAGES if not (FIGDIR / name).exists()]
    if missing:
        raise SystemExit(
            "Missing figures: " + ", ".join(missing) + "\nRun scripts/generate_figures.py first."
        )

    with PdfPages(OUT) as pdf:
        pdf.infodict()["Title"] = "Numerical Kitaev Chain Study - Figures"

        for name, title in PAGES:
            img = plt.imread(FIGDIR / name)
            fig = plt.figure(figsize=PAGESIZE)
            fig.text(0.5, 0.95, title, ha="center", va="center", fontsize=13)
            fig.text(
                0.5,
                0.035,
                r"$t=1$ everywhere; transition at $\mu=\pm2$",
                ha="center",
                va="center",
                fontsize=8,
                color="gray",
            )
            ax = fig.add_axes((0.04, 0.07, 0.92, 0.85))
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {OUT} ({len(PAGES)} pages)")


if __name__ == "__main__":
    main()
