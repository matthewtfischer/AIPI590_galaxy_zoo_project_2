"""Explainability experiments. Called from analysis.ipynb, which caches each result in results/.

Everything explains the model's log-odds for a class, log p(class) - log p(other),
since log-odds add up across features and probabilities near 0 or 1 don't.
"Removing" a band or region means setting it to zero, which after the arcsinh
stretch means no light there; the main model saw zeroed bands during training.
"""
import copy
import itertools
from math import factorial

import numpy as np
import pandas as pd
import torch
from captum.attr import KernelShap
from scipy.stats import spearmanr, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C, data as D


class LogOdds(torch.nn.Module):
    """Wraps a 2-class model so output column c is the log-odds of class c."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        z = self.model(x)
        d = z[:, 1] - z[:, 0]
        return torch.stack([-d, d], 1)


def split_index(data, split):
    return data.idx[split].cpu().numpy()


@torch.no_grad()
def predicted_class(model, x, batch_size=256):
    return torch.cat([model(x[i:i + batch_size]).argmax(1) for i in range(0, len(x), batch_size)])


@torch.no_grad()
def exact_shapley(model, x, target, n_players, keep_fn, batch_size=128):
    """Exact Shapley values for n_players groups of input features.

    keep_fn(x_batch, idx, subset) returns x_batch with every player not in
    `subset` removed. Returns (N, n_players) Shapley values of the target
    class log-odds, plus the (N, 2**n_players) table of log-odds for every
    subset and the list of subsets (tuples of 0/1) in the same column order.
    """
    f = LogOdds(model)
    subsets = list(itertools.product([0, 1], repeat=n_players))
    values = np.zeros((len(x), len(subsets)))
    for i in range(0, len(x), batch_size):
        xb, tb = x[i:i + batch_size], target[i:i + batch_size]
        idx = torch.arange(i, i + len(xb), device=x.device)
        for s, subset in enumerate(subsets):
            out = f(keep_fn(xb, idx, subset))
            values[i:i + len(xb), s] = out.gather(1, tb[:, None])[:, 0].cpu().numpy()
    code = {sub: k for k, sub in enumerate(subsets)}
    n = n_players
    phi = np.zeros((len(x), n))
    for p in range(n):
        for sub in subsets:
            if sub[p]:
                continue
            with_p = list(sub)
            with_p[p] = 1
            size = sum(sub)
            w = factorial(size) * factorial(n - size - 1) / factorial(n)
            phi[:, p] += w * (values[:, code[tuple(with_p)]] - values[:, code[sub]])
    return phi, values, subsets


def subset_name(subset):
    return "v_" + "".join(map(str, subset))


def shapley_table(phi, values, subsets, df, target, prefix):
    out = pd.DataFrame(phi, columns=[f"phi_{p}" for p in prefix])
    out.insert(0, "OBJID", df.OBJID.values)
    out["label"], out["pred"] = df.label.values, [C.CLASSES[t] for t in target.cpu().numpy()]
    n = len(prefix)
    out["v_all"] = values[:, subsets.index((1,) * n)]
    out["v_none"] = values[:, subsets.index((0,) * n)]
    for k, sub in enumerate(subsets):
        out[subset_name(sub)] = values[:, k]
    return out


def band_shapley(model, data, split, target=None):
    """Exact Shapley value of each of the five bands, per galaxy.

    By default each galaxy's predicted class is explained; pass `target` (class
    indices) to explain other classes instead.
    """
    ix = split_index(data, split)
    x = data.x[ix]
    target = predicted_class(model, x) if target is None else target

    def keep(xb, idx, subset):
        return xb * torch.tensor(subset, device=xb.device, dtype=xb.dtype)[None, :, None, None]

    phi, values, subsets = exact_shapley(model, x, target, len(C.BANDS), keep)
    return shapley_table(phi, values, subsets, data.df.iloc[ix], target, C.BANDS)


def region_shapley(model, data, split, target=None):
    """Exact Shapley value of each radial region (core, body, outskirts, background), all bands together.

    `target` works as in band_shapley.
    """
    ix = split_index(data, split)
    x = data.x[ix]
    target = predicted_class(model, x) if target is None else target
    regions = D.region_masks(data.df.iloc[ix], data.device)

    def keep(xb, idx, subset):
        keep_r = torch.tensor(subset, device=xb.device, dtype=xb.dtype)
        return xb * keep_r[regions[idx]][:, None]

    phi, values, subsets = exact_shapley(model, x, target, D.N_REGIONS, keep)
    return shapley_table(phi, values, subsets, data.df.iloc[ix], target, [f"region{r}" for r in range(D.N_REGIONS)])


def band_region_shap(model, data, split, n=200, n_samples=2000, seed=0):
    """KernelSHAP over the 20 band x region groups (too many for exact Shapley)."""
    rng = np.random.default_rng(seed)
    ix_all = split_index(data, split)
    df_all = data.df.iloc[ix_all].reset_index(drop=True)
    pick = np.concatenate([rng.choice(np.where(df_all.label == c)[0], n // 2, replace=False) for c in C.CLASSES])
    ix = ix_all[pick]
    x = data.x[ix]
    target = predicted_class(model, x)
    regions = D.region_masks(data.df.iloc[ix], data.device)
    ks = KernelShap(LogOdds(model))
    rows = []
    for j in range(len(ix)):
        fmask = (torch.arange(len(C.BANDS), device=x.device)[:, None, None] * D.N_REGIONS + regions[j][None])[None]
        torch.manual_seed(seed + j)
        a = ks.attribute(x[j:j + 1], baselines=0.0, target=target[j:j + 1], feature_mask=fmask,
                         n_samples=n_samples, perturbations_per_eval=250)[0]
        row = {"OBJID": data.df.OBJID.iloc[ix[j]], "label": data.df.label.iloc[ix[j]],
               "pred": C.CLASSES[int(target[j])]}
        for b, band in enumerate(C.BANDS):
            for r in range(D.N_REGIONS):
                m = regions[j] == r
                row[f"{band}_region{r}"] = float(a[b][m].mean()) if m.any() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def band_ablation(model, data, split="test"):
    """Accuracy with all bands, each single band alone, and each band left out."""
    from .train import metrics, predict
    rows = [{"bands": "all five", **metrics(*predict(model, data, split))}]
    for i, b in enumerate(C.BANDS):
        only = [int(j == i) for j in range(len(C.BANDS))]
        drop = [int(j != i) for j in range(len(C.BANDS))]
        rows.append({"bands": f"{b} only", **metrics(*predict(model, data, split, only))})
        rows.append({"bands": f"without {b}", **metrics(*predict(model, data, split, drop))})
    return pd.DataFrame(rows)


# --------------------------- checks on the Shapley values themselves ---------------------------

def randomized_copy(model, seed):
    """Copy of the model with every weight re-initialized (model randomization test)."""
    torch.manual_seed(seed)
    out = copy.deepcopy(model)
    for m in out.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return out.eval()


def per_galaxy_spearman(a, b):
    """Spearman correlation between two attribution vectors, one per galaxy (row)."""
    return np.array([spearmanr(x, y).statistic for x, y in zip(a, b)])


def deletion_curves(table, n_players, n_random=20, seed=0):
    """Log-odds of the predicted class as players are removed one at a time.

    Uses the log-odds already stored for every subset, so no new model calls are
    needed. Players are removed in three orders: largest Shapley value first,
    smallest first, and random (averaged over n_random orders). Returns one row
    per galaxy, order, and number removed.
    """
    rng = np.random.default_rng(seed)
    phi = table[[c for c in table.columns if c.startswith("phi_")]].values
    rows = []
    for i in range(len(table)):
        orders = {"largest first": np.argsort(-phi[i]), "smallest first": np.argsort(phi[i])}
        for name, order in orders.items():
            rows += _remove_in_order(table, i, order, n_players, name)
        for _ in range(n_random):
            rows += _remove_in_order(table, i, rng.permutation(n_players), n_players, "random")
    out = pd.DataFrame(rows)
    return out.groupby(["OBJID", "label", "order", "removed"], as_index=False).log_odds.mean()


def _remove_in_order(table, i, order, n_players, name):
    keep = [1] * n_players
    rows = []
    for k in range(n_players + 1):
        if k:
            keep[order[k - 1]] = 0
        rows.append({"OBJID": table.OBJID.iloc[i], "label": table.label.iloc[i], "order": name,
                     "removed": k, "log_odds": table[subset_name(tuple(keep))].iloc[i]})
    return rows


# --------------------------- catalog-feature baseline ---------------------------

CATALOG_FEATURES = {
    "color only (u − r, g − r)": ["u_minus_r", "g_minus_r"],
    "structure only (fracDeV_r, R90/R50)": ["fracDeV_r", "concentration"],
    "color and structure": ["u_minus_r", "g_minus_r", "fracDeV_r", "concentration"],
}


def catalog_baseline(df):
    """Logistic regression on SDSS catalog values, trained on the CNN's training split.

    Returns accuracy on the test and moderate-agreement sets for each feature
    set, plus the test-set predictions of the color and structure model.
    """
    df = df.assign(concentration=df.petroR90_r / df.petroR50_r)
    train = df[df.split == "train"]
    rows, preds = [], None
    for name, cols in CATALOG_FEATURES.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(train[cols], train.y)
        for split in ("test", "moderate"):
            d = df[df.split == split]
            rows.append({"features": name, "split": split, "acc": float((clf.predict(d[cols]) == d.y).mean())})
        if name == "color and structure":
            d = df[df.split == "test"]
            preds = pd.DataFrame({"OBJID": d.OBJID.values, "catalog_pred": clf.predict(d[cols])})
    return pd.DataFrame(rows), preds


# --------------------------- TCAV with Galaxy Zoo 2 concepts ---------------------------

def concept_sets(df):
    """Galaxies showing each concept, by Galaxy Zoo 2 debiased vote fractions.

    GZ2 only asks about bars, central bulges, and arms after a volunteer
    answers "features or disk" and then "not edge-on", so those fractions are
    conditional and can be high from one or two votes. Each concept also
    requires the earlier answers.
    """
    disk = df.gz2_features > 0.5
    face_on = disk & (df.gz2_not_edgeon > 0.5)
    return {
        "spiral arms": face_on & (df.gz2_arms > 0.9),
        "bar": face_on & (df.gz2_bar > 0.5),
        "prominent bulge": face_on & ((df.gz2_bulge_obvious + df.gz2_bulge_dominant) > 0.7),
        "edge-on disk": disk & (df.gz2_edgeon > 0.9),
        "round and smooth": (df.gz2_smooth > 0.8) & (df.gz2_round > 0.7),
        "tightly wound arms": face_on & (df.gz2_arms > 0.8) & (df.gz2_tight > 0.6),
        "loosely wound arms": face_on & (df.gz2_arms > 0.8) & (df.gz2_loose > 0.5),
    }


def tcav(model, data, layer=6, n_per_set=100, runs=20, seed=0):
    """TCAV scores for each concept and target class, against random concept sets.

    Concept and random example images come from the training and validation
    splits; scores are measured on test galaxies of the target class. A score is
    the fraction of those galaxies whose class log-odds increase when the layer's
    activations move in the concept's direction.
    """
    rng = np.random.default_rng(seed)
    bottom, top = model.split_at(layer)
    f = LogOdds(top)
    pool_ix = np.concatenate([split_index(data, "train"), split_index(data, "val")])
    pool_df = data.df.iloc[pool_ix].reset_index(drop=True)

    with torch.no_grad():
        acts = torch.cat([bottom(data.x[pool_ix[i:i + 256]]).flatten(1).cpu() for i in range(0, len(pool_ix), 256)]).numpy()

    test_ix = split_index(data, "test")
    grads = {}
    for t, cls in enumerate(C.CLASSES):
        ix = test_ix[data.df.y.values[test_ix] == t]
        gs = []
        for i in range(0, len(ix), 64):
            a = bottom(data.x[ix[i:i + 64]]).detach().requires_grad_(True)
            f(a)[:, t].sum().backward()
            gs.append(a.grad.flatten(1).cpu())
        grads[cls] = torch.cat(gs).numpy()

    def cav(pos, neg):
        X = np.concatenate([acts[pos], acts[neg]])
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(X, y)
        v = clf.coef_[0]
        return v / np.linalg.norm(v)

    def scores(v):
        return {cls: float((g @ v > 0).mean()) for cls, g in grads.items()}

    concepts = concept_sets(pool_df)
    rows = []
    for name, mask in concepts.items():
        pos_pool = np.where(mask.values)[0]
        rand_pool = np.where(~mask.values)[0]
        if len(pos_pool) < 30:
            continue
        for run in range(runs):
            pos = rng.choice(pos_pool, min(n_per_set, len(pos_pool)), replace=False)
            neg = rng.choice(rand_pool, n_per_set, replace=False)
            for cls, s in scores(cav(pos, neg)).items():
                rows.append({"concept": name, "run": run, "target": cls, "score": s, "n_concept_images": len(pos_pool)})
    for run in range(runs):
        a, b = rng.choice(len(pool_df), (2, n_per_set), replace=False)
        for cls, s in scores(cav(a, b)).items():
            rows.append({"concept": "random", "run": run, "target": cls, "score": s, "n_concept_images": n_per_set})
    return pd.DataFrame(rows)


def tcav_summary(df):
    rows = []
    for (concept, target), g in df[df.concept != "random"].groupby(["concept", "target"]):
        rand = df[(df.concept == "random") & (df.target == target)].score
        p = ttest_ind(g.score, rand, equal_var=False).pvalue
        rows.append({"concept": concept, "target": target, "mean_score": g.score.mean(), "std": g.score.std(),
                     "random_mean": rand.mean(), "p_value": p, "n_concept_images": g.n_concept_images.iloc[0]})
    out = pd.DataFrame(rows)
    out["significant"] = out.p_value < 0.05 / len(out)
    return out


# --------------------------- physical checks ---------------------------

def shares(phi):
    """Each player's share of the total absolute attribution, per galaxy."""
    a = np.abs(phi)
    return a / (a.sum(1, keepdims=True) + 1e-12)


def correlate(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    r = spearmanr(x[ok], y[ok])
    return r.statistic, r.pvalue, int(ok.sum())


def galaxy_builder_table():
    """Galaxy Builder fits, one row per galaxy (validation repeats averaged in)."""
    gb = pd.read_csv(C.GALAXY_BUILDER)
    fractions = ["bulge_fraction", "bar_fraction", "spiral_fraction"]
    # a component missing from the fitted model contributes no light
    gb[fractions] = gb[fractions].fillna(0.0)
    cols = fractions + ["disk_Re", "bulge_Re", "n_spirals"]
    return gb.groupby("dr7objid")[cols].mean().reset_index().rename(columns={"dr7objid": "OBJID"})
