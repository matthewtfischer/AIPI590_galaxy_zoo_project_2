import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from . import data as D
from .model import build

# Settings tuned with Optuna in project 1 (same task and architecture, but 3 input channels), reused here.
HPARAMS = {"width": 32, "dropout": 0.4636, "lr": 1.404e-4, "weight_decay": 0.01568, "batch_size": 64}

# band_dropout  each band zeroed with probability 0.15 during training
# plain         no band dropout
MODES = {"band_dropout": 0.15, "plain": 0.0}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at(epoch, base_lr, total, warmup=5):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * min(1.0, t)))


@torch.no_grad()
def predict(model, data, split, band_mask=None, batch_size=256):
    """p(elliptical) and labels for a split; band_mask (length 5, 0/1) blanks bands."""
    model.eval()
    ps, ys = [], []
    for _, x, y in data.batches(split, batch_size):
        if band_mask is not None:
            x = x * torch.as_tensor(band_mask, device=x.device, dtype=x.dtype)[None, :, None, None]
        ps.append(F.softmax(model(x), 1)[:, 1].cpu())
        ys.append(y.cpu())
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


def metrics(p, y):
    eps = 1e-7
    return {
        "acc": float(((p > 0.5) == y).mean()),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "nll": float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))),
    }


def fit(data, hp, p_band_dropout=0.0, epochs=150, patience=25, seed=590, verbose=True):
    """Train one model; returns (model at best validation loss, history)."""
    seed_everything(seed)
    g = torch.Generator(device=data.device).manual_seed(seed)
    model = build(hp).to(data.device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    best_loss, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        t0 = time.time()
        for group in opt.param_groups:
            group["lr"] = lr_at(ep, hp["lr"], epochs)
        model.train()
        total, correct, n = 0.0, 0, 0
        for _, x, y in data.batches("train", hp["batch_size"], shuffle=True, generator=g):
            x = D.random_d4(x, g)
            if p_band_dropout > 0:
                x = D.band_dropout(x, p_band_dropout, g)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)

        val = metrics(*predict(model, data, "val"))
        hist.append({"epoch": ep + 1, "lr": opt.param_groups[0]["lr"], "train_loss": total / n,
                     "train_acc": correct / n, "val_loss": val["nll"], "val_acc": val["acc"],
                     "sec": time.time() - t0})
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            h = hist[-1]
            print(f"epoch {ep + 1}: train {h['train_loss']:.4f} / {h['train_acc']:.3f}, "
                  f"val {h['val_loss']:.4f} / {h['val_acc']:.3f}, {h['sec']:.0f}s", flush=True)

        if val["nll"] < best_loss - 1e-4:
            best_loss, bad = val["nll"], 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"stopping at epoch {ep + 1}, best val loss {best_loss:.4f}")
                break

    model.load_state_dict(best_state)
    return model.eval(), hist
