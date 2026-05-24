# plot_utils.py
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from pathlib import Path

# Document constants
TEXTWIDTH = 6.30  # inches, from \the\textwidth in the LaTeX doc

# Load style once
plt.style.use(str(Path(__file__).parent / "paper.mplstyle"))

def figsize(fraction=0.8, ratio=0.62):
    w = TEXTWIDTH * fraction
    return (w, w * ratio)
