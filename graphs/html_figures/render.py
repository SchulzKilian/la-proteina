"""
Render HTML/CSS designs to PDFs for LaTeX inclusion.

Each figure lives under input/<name>/index.html (or input/<name>.html for a
flat single-file figure). Output goes to output/<name>.pdf. By default skips
figures whose PDF already exists; pass --regenerate to force re-render.

Style: paper.css is injected as the LAST stylesheet so it wins over the
source HTML's CSS — font becomes Latin Modern Roman to match paper.mplstyle.

First-run setup:
    pip install playwright
    playwright install chromium

Latin Modern OTFs are auto-downloaded from GUST on first render and cached
in fonts/. If the download fails, drop the OTFs in fonts/ manually
(see LM_FONTS_NEEDED below).

Usage:
    python render.py                  # render new figures only
    python render.py --regenerate     # rebuild all
    python render.py --only foo bar   # rebuild a subset
    python render.py --fraction 0.8   # 80% of \\textwidth (global default)
"""

import argparse
import asyncio
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

THIS_DIR = Path(__file__).parent
INPUT_DIR = THIS_DIR / "input"
OUTPUT_DIR = THIS_DIR / "output"
FONTS_DIR = THIS_DIR / "fonts"
PAPER_CSS = THIS_DIR / "paper.css"

# Matches graphs/plot_utils.py TEXTWIDTH — change both together.
TEXTWIDTH_IN = 6.30

# Latin Modern OTF distribution. CTAN mirror auto-redirects to a working
# mirror; the script finds the OTFs we need by basename anywhere in the zip.
LM_ZIP_URL = "https://mirrors.ctan.org/fonts/lm.zip"
LM_FONTS_NEEDED = [
    "lmroman10-regular.otf",
    "lmroman10-bold.otf",
    "lmroman10-italic.otf",
    "lmroman10-bolditalic.otf",
    "lmmono10-regular.otf",
    "lmmono10-italic.otf",
]


def ensure_fonts() -> None:
    """Download Latin Modern OTFs on first run, cache in fonts/."""
    FONTS_DIR.mkdir(exist_ok=True)
    missing = [f for f in LM_FONTS_NEEDED if not (FONTS_DIR / f).exists()]
    if not missing:
        return
    print(f"Fetching Latin Modern fonts from {LM_ZIP_URL}", flush=True)
    try:
        with urllib.request.urlopen(LM_ZIP_URL, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        sys.exit(
            f"Font download failed: {e}\n"
            f"Manually download {LM_ZIP_URL}, extract these files into "
            f"{FONTS_DIR}, and re-run:\n  " + "\n  ".join(missing)
        )
    print(f"  ({len(data) / 1e6:.1f} MB)", flush=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        members_by_name = {Path(n).name: n for n in zf.namelist() if n.endswith(".otf")}
        for f in missing:
            if f not in members_by_name:
                print(f"  WARNING: {f} not found in zip", file=sys.stderr)
                continue
            with zf.open(members_by_name[f]) as src, open(FONTS_DIR / f, "wb") as dst:
                dst.write(src.read())
            print(f"  installed {f}", flush=True)


def load_paper_css() -> str:
    """Read paper.css and rewrite relative font URLs to absolute file:// URIs.

    add_style_tag injects CSS into the page's document; relative URLs in
    that CSS resolve against the page URL, not paper.css's location, so
    we substitute absolute URIs before injection.
    """
    css = PAPER_CSS.read_text()
    fonts_uri = FONTS_DIR.absolute().as_uri()  # e.g. file:///.../fonts
    return css.replace('url("fonts/', f'url("{fonts_uri}/')


def collect_figures(only=None):
    """Yield (name, html_path) pairs from input/."""
    figures = []
    for entry in sorted(INPUT_DIR.iterdir()):
        if entry.is_dir():
            html = entry / "index.html"
            if html.exists():
                figures.append((entry.name, html))
            else:
                print(f"  skip {entry.name}/ (no index.html)", file=sys.stderr)
        elif entry.suffix.lower() == ".html":
            figures.append((entry.stem, entry))
    if only:
        wanted = set(only)
        figures = [(n, p) for (n, p) in figures if n in wanted]
        missing = wanted - {n for (n, _) in figures}
        for m in missing:
            print(f"  WARNING: --only {m} not found in input/", file=sys.stderr)
    return figures


# Minimal injection for native-font mode: white background, no margins, and
# neutralise full-viewport wrappers — but DON'T override font-family. Forcing
# Latin Modern changes per-glyph widths, which reflows text and can make
# carefully-spaced labels collide even though the design is fine in-browser.
NATIVE_CSS = (
    "html,body{margin:0!important;padding:0!important;background:white!important;"
    "min-height:0!important;height:auto!important}"
)


async def render_one(browser, name: str, html_path: Path, out_path: Path,
                     default_fraction: float, paper_css: str,
                     native_default: bool = False) -> None:
    """Render one HTML file to PDF, sized to fraction × \\textwidth."""
    page = await browser.new_page()
    try:
        await page.goto(html_path.absolute().as_uri(), wait_until="load")
        # Per-figure width override via <meta name="paper-fraction" content="0.8">.
        meta_fraction = await page.evaluate(
            "() => { const m = document.querySelector('meta[name=paper-fraction]');"
            " return m ? parseFloat(m.content) : null; }"
        )
        fraction = meta_fraction if meta_fraction is not None else default_fraction
        width_in = TEXTWIDTH_IN * fraction

        # Set viewport so layout reflows at the target print width before we measure.
        viewport_w_px = max(1, int(round(width_in * 96)))
        await page.set_viewport_size({"width": viewport_w_px, "height": 1080})

        # Per-figure font mode via <meta name="paper-font" content="off|on">.
        # "off" => keep the design's own font (no LM override); "on" => force LM.
        # Absent => fall back to the global --native-font default.
        meta_font = await page.evaluate(
            "() => { const m = document.querySelector('meta[name=paper-font]');"
            " return m ? m.content.trim().toLowerCase() : null; }"
        )
        if meta_font in ("off", "native", "false"):
            native = True
        elif meta_font in ("on", "paper", "true"):
            native = False
        else:
            native = native_default

        # Inject LAST so it overrides whatever the source declares. In paper-font
        # mode that's the full LM typography override; in native mode it's only
        # the background/crop reset, leaving the design's own fonts intact.
        await page.add_style_tag(content=NATIVE_CSS if native else paper_css)

        # Let any web-font / late layout settle.
        try:
            await page.evaluate("() => document.fonts && document.fonts.ready")
        except Exception:
            pass
        await page.wait_for_timeout(150)

        # Claude designs (and many handcrafted ones) wrap their figure in a
        # viewport-tall centering container — `min-height: 100vh` / `100dvh`
        # on body, .page, .container, whatever. paper.css can't anticipate
        # the class names, so neutralise structurally: walk the DOM, find any
        # element whose computed min-height/height equals (≈) the viewport,
        # and force it to fit content.
        await page.evaluate(
            """() => {
                const vh = window.innerHeight;
                const near = (px) => Math.abs(parseFloat(px) - vh) < 1;
                for (const el of document.querySelectorAll('*')) {
                  const cs = getComputedStyle(el);
                  if (near(cs.minHeight)) el.style.minHeight = '0';
                  if (near(cs.height))    el.style.height    = 'auto';
                }
            }"""
        )
        # Layout may have collapsed — give it a beat to settle before measuring.
        await page.wait_for_timeout(50)

        # Measure the actual content extent: the largest bottom edge across
        # all visible elements in the document. More robust than scrollHeight
        # when a wrapper forces min-height to the viewport, because we look
        # at where real layout content ends, not where the viewport ends.
        height_px = await page.evaluate(
            """() => {
                let maxBottom = 0;
                for (const el of document.body.querySelectorAll('*')) {
                  const r = el.getBoundingClientRect();
                  if (r.width === 0 || r.height === 0) continue;
                  const cs = getComputedStyle(el);
                  if (cs.visibility === 'hidden' || cs.display === 'none') continue;
                  const b = r.bottom + window.scrollY;
                  if (b > maxBottom) maxBottom = b;
                }
                if (maxBottom === 0) {
                  maxBottom = Math.max(
                    document.body.scrollHeight,
                    document.documentElement.scrollHeight
                  );
                }
                return maxBottom;
            }"""
        )
        height_in = max(height_px, 1) / 96

        await page.pdf(
            path=str(out_path),
            width=f"{width_in}in",
            height=f"{height_in}in",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            prefer_css_page_size=False,
        )
    finally:
        await page.close()
    outline_fonts(out_path)


def outline_fonts(pdf_path: Path) -> None:
    """Flatten all text to vector outlines via Ghostscript (-dNoOutputFonts).

    Chromium embeds the figure's text (esp. base64 @font-face web fonts) as
    Type 3 fonts. PDF.js — which both Overleaf's preview and macOS Preview use
    — rasterises Type 3 glyphs to cached bitmaps, so they look blurry on screen
    even though the data is vector. Converting text to outline paths removes all
    fonts from the PDF, so every viewer renders crisp vector shapes. Text is no
    longer selectable, which is fine for a figure. No-op if gs is unavailable.
    """
    import shutil
    import subprocess

    gs = shutil.which("gs")
    if gs is None:
        print("    (skip outline: ghostscript 'gs' not found — "
              "brew install ghostscript to de-Type3 the PDF)", file=sys.stderr)
        return
    tmp = pdf_path.with_suffix(".outlined.pdf")
    try:
        subprocess.run(
            [gs, "-o", str(tmp), "-dNoOutputFonts", "-sDEVICE=pdfwrite",
             str(pdf_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        tmp.replace(pdf_path)
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        print(f"    (outline failed: {e}; keeping un-outlined PDF)", file=sys.stderr)


async def main_async(args) -> None:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        sys.exit(
            "playwright is not installed. Install it once:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        )

    ensure_fonts()

    INPUT_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    figures = collect_figures(args.only)
    if not figures:
        print(f"No figures found in {INPUT_DIR}.")
        return

    todo = []
    for name, html in figures:
        out = OUTPUT_DIR / f"{name}.pdf"
        if out.exists() and not args.regenerate:
            print(f"  skip   {name}.pdf (exists)")
            continue
        todo.append((name, html, out))

    if not todo:
        print("Nothing to render. Pass --regenerate to rebuild existing PDFs.")
        return

    paper_css = load_paper_css()

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            for name, html, out in todo:
                print(f"  render {name}.pdf", flush=True)
                await render_one(browser, name, html, out, args.fraction,
                                 paper_css,
                                 native_default=not args.force_paper_font)
        finally:
            await browser.close()
    print(f"Done — {len(todo)} figure(s) → {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--regenerate", action="store_true",
                        help="re-render figures even if PDF exists")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="render only these figure names")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="default width as fraction of \\textwidth"
                             " (per-figure override via <meta name='paper-fraction'>)")
    parser.add_argument("--force-paper-font", action="store_true",
                        help="force Latin Modern on all figures (thesis-font"
                             " match). DEFAULT is native: keep each figure's own"
                             " font, since forcing LM reflows text and can make"
                             " carefully-spaced labels collide. Per-figure"
                             " override either way via <meta name='paper-font'>.")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
