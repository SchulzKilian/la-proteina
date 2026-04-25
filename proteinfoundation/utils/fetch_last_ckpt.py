import os
from typing import Union


def fetch_last_ckpt(ckpt_dir: str) -> Union[str, None]:
    """
    Returns the filename of the most recently written resumable checkpoint in
    ``ckpt_dir`` (picked by mtime). Considers both ``last*.ckpt`` (periodic
    snapshots controlled by ``last_ckpt_every_n_steps``) and ``best_val_*.ckpt``
    (end-of-validation top-k snapshots) — whichever is newest on disk.

    Rationale: picking the highest-numbered ``last-v<N>.ckpt`` can silently
    skip past newer ``best_val_*.ckpt`` files if a run is killed between
    ``last_ckpt_every_n_steps`` saves (see job 28260296 which lost ~450 steps
    by resuming from a ``last-v1.ckpt`` that was 3+ hours older than the
    newest ``best_val`` ckpt).

    Excludes ``-EMA.ckpt`` companions (loaded automatically by the EMA
    callback once the main ckpt is picked) and any ``ignore*.ckpt`` files
    produced by the periodic ``checkpoint_every_n_steps`` callback.
    """
    if not os.path.exists(ckpt_dir):
        return None
    candidates = [
        f
        for f in os.listdir(ckpt_dir)
        if f.endswith(".ckpt")
        and not f.endswith("-EMA.ckpt")
        and (f.startswith("last") or f.startswith("best_val_"))
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda f: os.path.getmtime(os.path.join(ckpt_dir, f)),
        reverse=True,
    )
    return candidates[0]
