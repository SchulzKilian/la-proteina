"""E100 analysis: sequence-diversity metrics vs latent-channel SDE jitter.

Reads the per-condition sample dirs produced by run_sde_jitter_entropy_probe.sh
(results/sde_jitter_entropy_probe/local<X>/samples/*.pt), computes per-sequence
diversity metrics, and reports per-condition means. Also cross-checks against a
subsample of the existing 1000-protein baseline panel and the AFDB-natural
reference, so the numbers are comparable to E099.

Metrics (per sequence, then averaged):
  shannon_bits      Shannon entropy of the 20-AA distribution (bits; max=log2(20)=4.32)
  max_aa_freq       frequency of the single most common residue
  longest_run       longest homopolymer run
  uniq_3mer         fraction of unique 3-mers among all 3-mer windows
  lowcplx_frac      fraction of residues in a low-complexity 12-window (<2.0 bits)
  net_charge        (#K+#R) - (#D+#E)
  net_charge_perL   net_charge / length
"""
from __future__ import annotations
import argparse, glob, math, os
from collections import Counter
import torch

AAS = "ACDEFGHIKLMNPQRSTVWY"


def shannon_bits(seq: str) -> float:
    n = len(seq)
    if n == 0:
        return 0.0
    c = Counter(seq)
    h = 0.0
    for a in AAS:
        p = c.get(a, 0) / n
        if p > 0:
            h -= p * math.log2(p)
    return h


def longest_run(seq: str) -> int:
    best = cur = 0
    prev = None
    for a in seq:
        cur = cur + 1 if a == prev else 1
        prev = a
        best = max(best, cur)
    return best


def uniq_3mer(seq: str) -> float:
    if len(seq) < 3:
        return 1.0
    kmers = [seq[i:i+3] for i in range(len(seq) - 2)]
    return len(set(kmers)) / len(kmers)


def lowcplx_frac(seq: str, w: int = 12, thr: float = 2.0) -> float:
    n = len(seq)
    if n < w:
        return float(shannon_bits(seq) < thr)
    flagged = [False] * n
    for i in range(n - w + 1):
        if shannon_bits(seq[i:i+w]) < thr:
            for j in range(i, i + w):
                flagged[j] = True
    return sum(flagged) / n


def net_charge(seq: str) -> int:
    c = Counter(seq)
    return (c.get("K", 0) + c.get("R", 0)) - (c.get("D", 0) + c.get("E", 0))


def load_seqs(samples_dir: str) -> list[tuple[str, int]]:
    out = []
    for p in sorted(glob.glob(os.path.join(samples_dir, "*.pt"))):
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        seq = d.get("sequence")
        if seq:
            out.append((seq, len(seq)))
    return out


def summarize(seqs: list[tuple[str, int]]) -> dict:
    if not seqs:
        return {}
    n = len(seqs)
    agg = {k: 0.0 for k in
           ["shannon_bits", "max_aa_freq", "longest_run", "uniq_3mer",
            "lowcplx_frac", "net_charge", "net_charge_perL", "length"]}
    comp = Counter()
    total_res = 0
    for seq, L in seqs:
        c = Counter(seq)
        agg["shannon_bits"] += shannon_bits(seq)
        agg["max_aa_freq"] += max(c.values()) / L
        agg["longest_run"] += longest_run(seq)
        agg["uniq_3mer"] += uniq_3mer(seq)
        agg["lowcplx_frac"] += lowcplx_frac(seq)
        nc = net_charge(seq)
        agg["net_charge"] += nc
        agg["net_charge_perL"] += nc / L
        agg["length"] += L
        comp.update(seq)
        total_res += L
    res = {k: v / n for k, v in agg.items()}
    res["n"] = n
    top = comp.most_common(3)
    res["top_aa"] = ", ".join(f"{a}={100*cnt/total_res:.1f}%" for a, cnt in top)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/sde_jitter_entropy_probe")
    ap.add_argument("--baseline_panel",
                    default="results/generated_stratified_300_800_nsteps400/samples")
    args = ap.parse_args()

    conds = sorted(glob.glob(os.path.join(args.root, "local*")))
    rows = []
    for c in conds:
        sd = os.path.join(c, "samples")
        seqs = load_seqs(sd)
        s = summarize(seqs)
        if s:
            s["label"] = os.path.basename(c)
            rows.append(s)

    # Cross-check: subsample (first 60) of the existing 1000-panel
    base_seqs = load_seqs(args.baseline_panel)[:60]
    if base_seqs:
        s = summarize(base_seqs)
        s["label"] = "1000panel(n<=60)"
        rows.append(s)

    print("\n=== E100: sequence diversity vs latent-channel SDE jitter ===")
    print("(matched seed/length across local0.05/0.15/0.30; bb_ca fixed 0.15; nsteps=400)\n")
    hdr = f"{'condition':18}{'n':>4}{'shannon':>9}{'maxAA%':>8}{'run':>6}{'uniq3mer':>9}{'lowcplx':>9}{'netQ':>7}{'Q/L':>8}  top_aa"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:18}{r['n']:>4}{r['shannon_bits']:>9.3f}"
              f"{100*r['max_aa_freq']:>8.1f}{r['longest_run']:>6.1f}"
              f"{r['uniq_3mer']:>9.3f}{r['lowcplx_frac']:>9.3f}"
              f"{r['net_charge']:>7.1f}{r['net_charge_perL']:>8.3f}  {r['top_aa']}")
    print("\nReference: AFDB-natural Shannon (property panel, E099) = 4.05 bits; "
          "max for 20 AA = log2(20) = 4.32 bits.")
    print("NOTE: Shannon here is computed in BITS directly from the .pt sequence; "
          "the E099 3.47-vs-4.05 gap used the developability-panel shannon_entropy "
          "column. Cross-check the 1000panel row against the local0.05 row — both "
          "should land near each other if 0.05 reproduces the baseline.")


if __name__ == "__main__":
    main()
