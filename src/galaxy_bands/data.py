import json

import numpy as np
import pandas as pd
import torch

from . import config as C

CUBE_CACHE = C.DATA / "cubes.npy"
SCALES_JSON = C.RESULTS / "band_noise.json"
MIN_COVERAGE = 0.99
N_REGIONS = 4
REGION_NAMES = ["core (< R50)", "body (R50-R90)", "outskirts (R90-1.5 R90)", "background"]


def load_sample():
    """Sample table restricted to galaxies whose cutout is fully covered in all five bands."""
    df = pd.read_csv(C.SAMPLE_CSV)
    st = pd.read_csv(C.CUTOUT_STATUS)
    st["coverage"] = st[[f"coverage_{b}" for b in C.BANDS]].min(axis=1)
    df = df.merge(st[["OBJID", "coverage"]], on="OBJID")
    df = df[df.coverage >= MIN_COVERAGE].reset_index(drop=True)
    df["r50_pix"] = df.petroR50_r / df.scale
    df["r90_pix"] = df.petroR90_r / df.scale
    df["u_minus_r"] = df.dered_u - df.dered_r
    df["g_minus_r"] = df.dered_g - df.dered_r
    return df


def load_cubes(df):
    """Raw fluxes (nanomaggies), shape (N, 5, 128, 128), cached after the first load."""
    if CUBE_CACHE.exists():
        cubes = np.load(CUBE_CACHE, mmap_mode="r")
        ids = np.load(CUBE_CACHE.with_suffix(".ids.npy"))
        if np.array_equal(ids, df.OBJID.values):
            return np.asarray(cubes)
    cubes = np.stack([np.load(C.CUTOUT_DIR / f"{o}.npy") for o in df.OBJID])
    np.save(CUBE_CACHE, cubes)
    np.save(CUBE_CACHE.with_suffix(".ids.npy"), df.OBJID.values)
    return cubes


def band_noise(cubes, df):
    """Typical background noise per band, from the corners of the training cutouts (MAD estimate)."""
    if SCALES_JSON.exists():
        return np.array(json.loads(SCALES_JSON.read_text()))
    yy, xx = np.mgrid[:C.IMG_SIZE, :C.IMG_SIZE]
    corner = np.hypot(yy - (C.IMG_SIZE - 1) / 2, xx - (C.IMG_SIZE - 1) / 2) > 56
    train = cubes[(df.split == "train").values][:, :, corner]
    med = np.median(train, axis=2, keepdims=True)
    sigma = 1.4826 * np.median(np.abs(train - med), axis=2)
    noise = np.median(sigma, axis=0)
    C.RESULTS.mkdir(exist_ok=True)
    SCALES_JSON.write_text(json.dumps(noise.tolist()))
    return noise


def normalize(cubes, noise):
    """arcsinh stretch of each band in units of its own background noise."""
    return np.arcsinh(cubes / noise[None, :, None, None]).astype(np.float32)


def region_masks(df, device):
    """Integer map per galaxy: 0 core, 1 body, 2 outskirts, 3 background."""
    g = torch.arange(C.IMG_SIZE, device=device, dtype=torch.float32) - (C.IMG_SIZE - 1) / 2
    r = torch.hypot(g[:, None], g[None, :])[None]
    r50 = torch.tensor(df.r50_pix.values, device=device, dtype=torch.float32)[:, None, None]
    r90 = torch.tensor(df.r90_pix.values, device=device, dtype=torch.float32)[:, None, None]
    regions = torch.full((len(df), C.IMG_SIZE, C.IMG_SIZE), 3, device=device, dtype=torch.long)
    regions[r < 1.5 * r90] = 2
    regions[r < r90] = 1
    regions[r < r50] = 0
    return regions


class GalaxyData:
    def __init__(self, device):
        self.df = load_sample()
        cubes = load_cubes(self.df)
        self.noise = band_noise(cubes, self.df)
        self.x = torch.from_numpy(normalize(cubes, self.noise)).to(device)
        self.y = torch.tensor(self.df.y.values, dtype=torch.long, device=device)
        self.idx = {s: torch.tensor(np.where(self.df.split.values == s)[0], device=device)
                    for s in ("train", "val", "test", "galaxy_builder", "moderate")}
        self.device = device

    def batches(self, split, batch_size, shuffle=False, generator=None):
        idx = self.idx[split]
        if shuffle:
            idx = idx[torch.randperm(len(idx), device=self.device, generator=generator)]
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            yield b, self.x[b], self.y[b]


def random_d4(x, generator=None):
    """Random 90 degree rotation and flip per image (exact, no interpolation)."""
    n = x.shape[0]
    k = torch.randint(0, 4, (n,), device=x.device, generator=generator)
    flip = torch.rand(n, device=x.device, generator=generator) < 0.5
    out = torch.empty_like(x)
    for kk in range(4):
        m = k == kk
        if m.any():
            out[m] = torch.rot90(x[m], kk, dims=(-2, -1))
    out[flip] = out[flip].flip(-1)
    return out


def band_dropout(x, p, generator=None):
    """Zero out each band with probability p, always keeping at least one band."""
    keep = torch.rand(x.shape[0], x.shape[1], device=x.device, generator=generator) >= p
    none = ~keep.any(1)
    if none.any():
        pick = torch.randint(0, x.shape[1], (int(none.sum()),), device=x.device, generator=generator)
        keep[none.nonzero()[:, 0], pick] = True
    return x * keep[:, :, None, None]
