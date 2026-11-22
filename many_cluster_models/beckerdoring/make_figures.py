#!/usr/bin/env python3
"""
Part4 figure set for the Becker-Doring (Part IV) two-pole plasmid model.

    python make_figures.py --all            # every figure  (cached SSA reused)
    python make_figures.py --checks         # numerical checks, no figures
    python make_figures.py --fig time_course imbalance_ssa
    python make_figures.py --all --quick    # fewer replicates, fast

Figures go to ./figures, SSA results to ./cache.  The model code is shared
with the other form in ../plasmid_common (plasmid.py, ssa.c, figures.py).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "plasmid_common"))

from figures import main  # noqa: E402

if __name__ == "__main__":
    main(model="bd", here=HERE)
