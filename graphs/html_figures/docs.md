# `graphs/html_figures/` — HTML/CSS → PDF figures

For schematics, flowcharts, conceptual diagrams etc. that are easier to design
in HTML/CSS (e.g. Claude design output) than in matplotlib. Renders to vector
PDFs whose typography matches the rest of the thesis, so they drop into
`\includegraphics{}` next to matplotlib figures with no font mismatch.

## Layout

```
graphs/html_figures/
├── render.py     — generator
├── paper.css     — typography overrides (mirror of paper.mplstyle)
├── fonts/        — Latin Modern OTFs (auto-downloaded on first run)
├── input/        — drop HTML here
│   └── my_diagram/
│       ├── index.html
│       └── (optional assets: images, CSS, JS)
└── output/       — rendered PDFs (one per figure)
```

Two input forms:
- **Folder:** `input/my_diagram/index.html` + any assets. → `output/my_diagram.pdf`.
- **Flat file:** `input/my_diagram.html`. → `output/my_diagram.pdf`.

## First-run setup

```bash
pip install playwright
playwright install chromium
brew install ghostscript   # outlines fonts post-render (see below); optional but recommended
```

Latin Modern Roman / Mono OTFs are downloaded from GUST e-foundry on the
first render and cached in `fonts/`. If the download fails, grab the OTFs
listed in `LM_FONTS_NEEDED` (top of `render.py`) and drop them in `fonts/`
manually.

## Generate

```bash
python graphs/html_figures/render.py                  # new figures only
python graphs/html_figures/render.py --regenerate     # force rebuild all
python graphs/html_figures/render.py --only foo bar   # subset
python graphs/html_figures/render.py --fraction 0.8   # 80% of textwidth
```

By default the script skips any figure whose PDF already exists — pass
`--regenerate` to force.

## Per-figure width

Default page width is `\textwidth = 6.30 in` (set in `render.py` to match
`graphs/plot_utils.TEXTWIDTH`). Override:

- **Globally:** `--fraction 0.5`.
- **Per figure:** add a meta tag in the HTML `<head>`:

```html
<meta name="paper-fraction" content="0.5">
```

Height is computed from the natural content height of the rendered page — so
design the HTML so the figure ends exactly where you want the PDF to end. The
script forces 0 page margins; control whitespace via CSS `padding` on `body`.

## Drop into LaTeX

```latex
\includegraphics{graphs/html_figures/output/my_diagram.pdf}
```

Vector PDF; fonts embedded.

## Fonts: native by default

**By default the figure keeps its own font** — `render.py` injects only a
background/crop reset, not the Latin Modern override. Forcing LM changes every
glyph's width, which reflows text and can make carefully-spaced labels collide
even though the design looks fine in-browser (observed on the pipeline/steering
figures). Native rendering is faithful: the PDF matches what you designed.

To match thesis body type instead (Latin Modern, at the cost of possible
reflow):

- **Globally:** `python render.py --force-paper-font`.
- **Per figure:** `<meta name="paper-font" content="on">` in the HTML `<head>`
  (`content="off"` forces native even under `--force-paper-font`).

## How style overrides work

Whatever CSS is injected goes in as the **last** stylesheet by `render.py`, so
it wins over the source HTML's CSS even when that CSS uses inline `style="…"` or
Tailwind utility classes. Under `--force-paper-font`, `paper.css` forces Latin
Modern Roman via `!important` on `*:not(code, pre, ...)`; code-like elements
stay in Latin Modern Mono.

CSS custom properties exposed for figures that want exact line widths from
`paper.mplstyle`:

```css
border: var(--paper-axis-width) solid black;   /* 0.5pt */
border: var(--paper-line-width) solid black;   /* 1.0pt */
border: var(--paper-grid-width) solid #888;    /* 0.4pt */
```

## Notes & limits

- **Math:** if a figure needs math, include MathJax or KaTeX in the HTML and
  point it at Latin Modern Math for visual consistency with the body text.
- **Type 3 fonts / blurry preview:** Chromium embeds the figure's text (esp.
  base64 `@font-face` web fonts) as **Type 3** fonts. PDF.js — used by both
  Overleaf's preview pane and macOS Preview — rasterises Type 3 glyphs to cached
  bitmaps, so they look blurry on screen even though the data is vector. After
  each render the script runs `gs -dNoOutputFonts` (Ghostscript) to flatten all
  text to vector outlines, leaving **zero fonts** in the PDF — crisp in every
  viewer. Text becomes non-selectable (fine for a figure). If `gs` is missing,
  rendering still works but the PDF keeps its Type 3 fonts (blurry in PDF.js);
  `brew install ghostscript` to fix.
- **Rasterised effects:** Chromium's PDF export rasterises some CSS effects
  (`box-shadow` with big radii, `filter: blur`, etc.). Vector text, SVG, and
  basic CSS shapes stay vector.
- **JS / dynamic layout:** the page is rendered after `load` + a short
  font-ready wait. Mermaid / d3 / static-after-load JS works; long
  animations need extra wait logic.
