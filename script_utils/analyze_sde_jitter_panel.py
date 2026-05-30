"""E100 panel comparison: developability properties vs latent-channel SDE jitter.

Reads results/sde_jitter_entropy_probe/local<X>/properties_generated.csv for
each condition and reports per-condition means for the properties E099 flagged,
plus a length-matched standardized gap vs AFDB-natural (same metric as E099).
"""
from __future__ import annotations
import argparse, csv, glob, math, os
from collections import defaultdict

PROPS = ["shannon_entropy", "net_charge_ph7", "swi", "tango_total",
         "sap_total", "scm_positive", "radius_of_gyration",
         "iupred3_fraction_disordered"]


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for d in csv.DictReader(f):
            rows.append(d)
    return rows


def fbin(L):
    L = int(float(L))
    return min([300, 400, 500, 600, 700, 800], key=lambda x: abs(x - L))


def ms(x):
    if not x:
        return float("nan"), float("nan")
    m = sum(x) / len(x)
    s = (sum((v - m) ** 2 for v in x) / max(1, len(x) - 1)) ** 0.5
    return m, s


def collect_bylen(rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            b = fbin(r["sequence_length"])
        except Exception:
            continue
        for p in PROPS:
            v = r.get(p, "")
            if v not in ("", "nan", "NaN", None):
                try:
                    by[p][b].append(float(v))
                except Exception:
                    pass
    return by


def cond_mean(rows, p):
    vals = []
    for r in rows:
        v = r.get(p, "")
        if v not in ("", "nan", "NaN", None):
            try:
                vals.append(float(v))
            except Exception:
                pass
    return ms(vals)[0]


def lenmatched_gap(gen_rows, ref_by):
    """Avg standardized gap (gen_mean - nat_mean)/nat_sd over shared L-bins."""
    G = collect_bylen(gen_rows)
    out = {}
    for p in PROPS:
        ds = []
        for b in [300, 400, 500, 600, 700, 800]:
            g = G[p].get(b)
            r = ref_by[p].get(b)
            if g and r and len(r) > 2:
                gm, _ = ms(g)
                nm, ns = ms(r)
                if ns > 0:
                    ds.append((gm - nm) / ns)
        out[p] = sum(ds) / len(ds) if ds else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/sde_jitter_entropy_probe")
    ap.add_argument("--afdb", default="data/afdb_ref/properties_afdb.csv")
    args = ap.parse_args()

    ref_by = collect_bylen(load(args.afdb))

    conds = sorted(glob.glob(os.path.join(args.root, "local*")))
    data = {}
    for c in conds:
        rows = load(os.path.join(c, "properties_generated.csv"))
        if rows:
            data[os.path.basename(c)] = rows

    print("\n=== E100: developability panel vs latent-channel SDE jitter ===")
    print("(bb_ca fixed 0.15; matched seed/length; nsteps=400)\n")

    # Per-condition raw means
    labels = list(data.keys())
    print(f"{'property':28}" + "".join(f"{l:>12}" for l in labels))
    print("-" * (28 + 12 * len(labels)))
    for p in PROPS:
        line = f"{p:28}"
        for l in labels:
            line += f"{cond_mean(data[l], p):>12.3f}"
        print(line)

    # Length-matched standardized gap vs natural (lower |.| = closer to natural)
    print("\n--- length-matched standardized gap vs AFDB-natural (d) ---")
    print(f"{'property':28}" + "".join(f"{l:>12}" for l in labels))
    print("-" * (28 + 12 * len(labels)))
    gaps = {l: lenmatched_gap(data[l], ref_by) for l in labels}
    for p in PROPS:
        line = f"{p:28}"
        for l in labels:
            line += f"{gaps[l][p]:>+12.2f}"
        print(line)

    print("\nReference E099 (1000-panel @ local0.05): shannon d=-5.36, "
          "net_charge d=-1.74, swi d=+1.34, tango d=-0.76, sap d=-1.27, "
          "scm_pos d=-0.84, Rg d=-0.90.")
    print("Interpretation: if raising local jitter moves shannon d toward 0 "
          "(and net_charge toward 0) it de-collapses sequences toward natural.")


if __name__ == "__main__":
    main()
