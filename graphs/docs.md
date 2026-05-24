# `graphs/` — plotting infrastructure

Self-contained matplotlib setup for the thesis. Two files do the work; every
plot script in this repo should use them so that every figure comes out with
identical typography to the LaTeX document.

## `paper.mplstyle`

A matplotlib rc-style file that configures the **PGF + LaTeX backend** to match
a **12pt LaTeX paper document**:

- `pgf.texsystem: pdflatex`, `text.usetex: true`, `font.family: serif`,
  `font.serif: Computer Modern Roman` — figures use the same CM Roman as the
  body text, so labels are visually indistinguishable from surrounding prose.
- Exact pt sizes: 12pt for axis labels and titles, 10pt for legends and tick
  labels.
- Thin axis / tick / grid linewidths (0.4–0.5pt) for a clean print look.
- Default save format is **PDF** (`savefig.format: pdf`), tight bbox, 0.02in
  padding.

## `plot_utils.py`

Small helper module. On import it:

1. Forces matplotlib to use the **PGF backend** (must happen before any
   `pyplot` use).
2. Loads `paper.mplstyle` automatically.
3. Exposes `figsize(fraction, ratio)`:
   - returns a `(width, height)` tuple in inches,
   - scaled relative to the LaTeX `\textwidth = 6.30 in` constant,
   - `fraction` = width as a fraction of `\textwidth` (e.g. `0.8` → 80% of
     textwidth, `1.0` → full width, `0.5` → half-width),
   - `ratio` = height / width (default `0.62`, golden-ratio-ish).

If `\textwidth` in the LaTeX source changes, update `TEXTWIDTH` in
`plot_utils.py`.

## Canonical starter snippet

Every plot script must begin with this:

```python
from plot_utils import figsize
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=figsize(0.8))
# ....
```

This guarantees the PGF backend is active, the style is loaded, and the figure
size is locked to the document width.

## Saving

```python
fig.savefig("results/myplot.pdf")
```

PDF is the default format from the style, so no `format=` argument is needed.
The output PDF drops straight into LaTeX via
`\includegraphics{graphs/results/myplot.pdf}` — no rasterisation, no font
mismatch.

## LaTeX math in labels

Because `text.usetex: True` is set, LaTeX math renders natively in any
matplotlib text. Use raw strings:

```python
ax.set_xlabel(r"$\sigma$")
ax.set_ylabel(r"$\log p(x)$")
ax.set_title(r"Effect of $w_{\max}$ on scRMSD")
```

Any LaTeX command available in `pdflatex` works (no need to load extra
packages for standard math).

## Output location

Save all PDFs into `graphs/results/`. The directory is reserved for figure
output so later agents and scripts have a consistent place to write to.
