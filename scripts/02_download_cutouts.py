"""Make 5-band (u, g, r, i, z) cutouts for every galaxy in data/sample.csv.

For each SDSS field the five calibrated frames are downloaded from the DR17
Science Archive Server (sky-subtracted images in nanomaggies), and each band
is resampled onto the same 128x128 grid centered on the galaxy. The grid is
4 * petroR90_r across, so R90 lands 32 px from the center, like project 1.
Resampling every band onto one grid also lines the bands up, since the five
frames of a field aren't pixel-aligned with each other.

Frames are only held in memory, so disk use stays small. Cutouts are saved as
data/cutouts/<OBJID>.npy (float32, shape 5 x 128 x 128, north up) and the
script skips any galaxy that already has one, so it can be rerun.

    python scripts/02_download_cutouts.py [--limit N] [--workers 8]
"""
import argparse
import bz2
import io
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galaxy_bands import config as C  # noqa: E402


def download_frame(rerun, run, camcol, field, band):
    url = C.SDSS_FRAME_URL.format(rerun=rerun, run=run, camcol=camcol, field=field, band=band)
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with fits.open(io.BytesIO(bz2.decompress(r.content))) as hdul:
                return hdul[0].data.astype(np.float32), WCS(hdul[0].header)
        except Exception as e:
            err = e
    raise RuntimeError(f"{url}: {err}")


def target_wcs(ra, dec, scale):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(C.IMG_SIZE + 1) / 2, (C.IMG_SIZE + 1) / 2]
    w.wcs.cdelt = [-scale / 3600, scale / 3600]
    return w


def process_field(key, galaxies):
    rerun, run, camcol, field = key
    frames = {b: download_frame(rerun, run, camcol, field, b) for b in C.BANDS}
    rows = []
    for g in galaxies:
        out = C.CUTOUT_DIR / f"{g['OBJID']}.npy"
        wcs_out = target_wcs(g["ra_deg"], g["dec_deg"], g["scale"])
        cube, coverage = [], []
        for b in C.BANDS:
            data, wcs_in = frames[b]
            img, foot = reproject_interp((data, wcs_in), wcs_out, shape_out=(C.IMG_SIZE, C.IMG_SIZE))
            cube.append(np.nan_to_num(img))
            coverage.append(float(foot.mean()))
        # FITS rows run south to north; flip so row 0 is north, like the JPEG cutouts
        np.save(out, np.flip(np.stack(cube), axis=1).astype(np.float32))
        rows.append({"OBJID": g["OBJID"], **{f"coverage_{b}": c for b, c in zip(C.BANDS, coverage)}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only process this many fields (for testing)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    C.CUTOUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.SAMPLE_CSV)
    todo = df[~df.OBJID.apply(lambda o: (C.CUTOUT_DIR / f"{o}.npy").exists())]
    groups = list(todo.groupby(["rerun", "run", "camcol", "field"]))
    if args.limit:
        groups = groups[:args.limit]
    print(f"{len(todo)} galaxies left in {len(groups)} fields")

    rows, failed = [], []
    with ProcessPoolExecutor(args.workers) as ex:
        futures = {ex.submit(process_field, key, g.to_dict("records")): key for key, g in groups}
        for f in tqdm(as_completed(futures), total=len(futures), mininterval=10):
            try:
                rows += f.result()
            except Exception as e:
                failed.append((futures[f], str(e)[:200]))

    status = pd.DataFrame(rows)
    if C.CUTOUT_STATUS.exists():
        status = pd.concat([pd.read_csv(C.CUTOUT_STATUS), status]).drop_duplicates("OBJID", keep="last")
    status.to_csv(C.CUTOUT_STATUS, index=False)
    print(f"done: {len(rows)} new cutouts, {len(failed)} fields failed")
    for key, msg in failed[:10]:
        print("  failed", key, msg)


if __name__ == "__main__":
    main()
