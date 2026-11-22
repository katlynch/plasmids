#!/usr/bin/env python3
"""
The katlynch/plasmids figure set (files at the root of this repository) for this model.

    python make_kat_figures.py --all
    python make_kat_figures.py --fig odesolve closure decay percent_free \
                                     eigenvalues pseudo_bif bifurcation pitchfork_locus

Figures go to ./katlynch_figures, cached results to ./cache.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plasmid_common"))

from kat_figures import main  # noqa: E402

if __name__ == "__main__":
    main(model="smol", here=HERE)
