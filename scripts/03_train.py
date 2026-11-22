"""Train the 5-band CNN twice: with band dropout (the main model) and without.

Up to 150 epochs with early stopping (patience 25) on validation loss.
Checkpoints go to models/<mode>.pt, learning curves to results/history_<mode>.csv
and test metrics to results/test_metrics.csv. Metrics cover the test set, the
Galaxy Builder galaxies and the moderate-agreement galaxies.

    python scripts/03_train.py [--modes band_dropout plain] [--epochs 150] [--eval-only]

--eval-only skips training and re-scores the saved checkpoints.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galaxy_bands import config as C, data as D, model as M, train as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=list(T.MODES))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    data = D.GalaxyData(T.get_device())
    print({s: len(i) for s, i in data.idx.items()})
    C.MODELS.mkdir(exist_ok=True)

    rows = []
    for mode in args.modes:
        print(f"--- {mode}")
        if args.eval_only:
            model = M.load(C.MODELS / f"{mode}.pt", data.device)
        else:
            model, hist = T.fit(data, T.HPARAMS, p_band_dropout=T.MODES[mode], epochs=args.epochs)
            pd.DataFrame(hist).to_csv(C.RESULTS / f"history_{mode}.csv", index=False)
            M.save(model, T.HPARAMS, C.MODELS / f"{mode}.pt", mode=mode, epochs_run=len(hist))
        for split in ("test", "galaxy_builder", "moderate"):
            rows.append({"model": mode, "split": split, **T.metrics(*T.predict(model, data, split))})
            print(rows[-1])

    out = C.RESULTS / "test_metrics.csv"
    if out.exists():
        old = pd.read_csv(out)
        rows = old[~old.model.isin(args.modes)].to_dict("records") + rows
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
