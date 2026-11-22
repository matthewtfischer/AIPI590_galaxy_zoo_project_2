"""Build the galaxy sample

The sample is made up of confident spiral and elliptical galaxies from Galaxy Zoo 1 (debiased vote
fraction above 0.85, at least 30 votes) that also appear in the Galaxy Zoo 2
catalog, so every galaxy has detailed morphology votes too. The Galaxy
Builder galaxies are added as a separate evaluation set and never used for
training.

A third evaluation set, also never used for training, has 500 galaxies of each
type where volunteers agreed less: 70-85% of the debiased vote went to the
winning type (split "moderate").

For each galaxy the SDSS DR17 database gives the imaging frame it sits in
(run, rerun, camcol, field), its Petrosian radii, extinction-corrected
magnitudes in all five bands, fracDeV_r, and the spectroscopic redshift.
GZ1/GZ2 IDs are DR7 IDs, which aren't in DR17, so the lookup matches on
position (nearest object within 3 arcsec).

Writes data/sample.csv. If it already exists, galaxies already in it are kept
as they are and only new ones are looked up.
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from galaxy_bands import config as C  # noqa: E402

BATCH = 100

GZ2_COLUMNS = {
    "t01_smooth_or_features_a01_smooth_debiased": "gz2_smooth",
    "t01_smooth_or_features_a02_features_or_disk_debiased": "gz2_features",
    "t02_edgeon_a04_yes_debiased": "gz2_edgeon",
    "t02_edgeon_a05_no_debiased": "gz2_not_edgeon",
    "t03_bar_a06_bar_debiased": "gz2_bar",
    "t03_bar_a07_no_bar_debiased": "gz2_no_bar",
    "t04_spiral_a08_spiral_debiased": "gz2_arms",
    "t05_bulge_prominence_a10_no_bulge_debiased": "gz2_bulge_none",
    "t05_bulge_prominence_a11_just_noticeable_debiased": "gz2_bulge_noticeable",
    "t05_bulge_prominence_a12_obvious_debiased": "gz2_bulge_obvious",
    "t05_bulge_prominence_a13_dominant_debiased": "gz2_bulge_dominant",
    "t07_rounded_a16_completely_round_debiased": "gz2_round",
    "t10_arms_winding_a28_tight_debiased": "gz2_tight",
    "t10_arms_winding_a30_loose_debiased": "gz2_loose",
}


def to_degrees(ra, dec):
    h, m, s = ra.str.split(":", expand=True).astype(float).T.values
    d = dec.str.split(":", expand=True)
    sign = np.where(d[0].str.strip().str.startswith("-"), -1, 1)
    dd, dm, ds = d.astype(float).abs().T.values
    return 15 * (h + m / 60 + s / 3600), sign * (dd + dm / 60 + ds / 3600)


def query_sdss(rows):
    values = ",".join(f"({r.OBJID},{r.ra_deg:.6f},{r.dec_deg:.6f})" for r in rows.itertuples())
    sql = ("select v.k as OBJID, p.run, p.rerun, p.camcol, p.field, p.petroR50_r, p.petroR90_r, p.petroMag_r, "
           "p.dered_u, p.dered_g, p.dered_r, p.dered_i, p.dered_z, p.fracDeV_r, s.z "
           f"from (values {values}) as v(k,ra,dec) "
           "cross apply dbo.fGetNearestObjEq(v.ra, v.dec, 0.05) as n "
           "join PhotoObj as p on p.objid = n.objid "
           "left join SpecObj as s on s.bestobjid = p.objid")
    for attempt in range(5):
        try:
            r = requests.get(C.SDSS_SQL_URL, params={"cmd": sql, "format": "csv"}, timeout=120)
            r.raise_for_status()
            return pd.read_csv(StringIO(r.text), skiprows=1)
        except Exception as e:
            err = e
    raise RuntimeError(f"SDSS query failed: {err}")


def main():
    gz1 = pd.read_csv(C.GZ1_TABLE)
    gz2 = pd.read_csv(C.GZ2_TABLE, usecols=["dr7objid", *GZ2_COLUMNS]).rename(columns=GZ2_COLUMNS)
    gb = pd.read_csv(C.GALAXY_BUILDER)

    gz1 = gz1[gz1.NVOTE >= C.MIN_VOTES].copy()
    gz1["label"] = np.select([gz1.P_CS_DEBIASED > C.CONF_THRESHOLD, gz1.P_EL_DEBIASED > C.CONF_THRESHOLD],
                             ["spiral", "elliptical"], default="")
    pool = gz1[(gz1.label != "") & gz1.OBJID.isin(gz2.dr7objid) & ~gz1.OBJID.isin(gb.dr7objid)]
    main_sample = pool.groupby("label").sample(C.N_PER_CLASS, random_state=C.SEED)

    lo, hi = C.MODERATE_RANGE
    eligible = gz1[gz1.OBJID.isin(gz2.dr7objid) & ~gz1.OBJID.isin(gb.dr7objid)].copy()
    eligible["label"] = np.select([eligible.P_CS_DEBIASED.between(lo, hi, inclusive="right"),
                                   eligible.P_EL_DEBIASED.between(lo, hi, inclusive="right")],
                                  ["spiral", "elliptical"], default="")
    moderate = eligible[eligible.label != ""].groupby("label").sample(C.N_MODERATE_PER_CLASS, random_state=C.SEED)
    moderate["split"] = "moderate"

    gb_ids = gb.dr7objid.drop_duplicates()
    gb_sample = pd.read_csv(C.GZ1_TABLE)
    gb_sample = gb_sample[gb_sample.OBJID.isin(gb_ids)].copy()
    gb_sample["label"] = "spiral"

    main_sample["split"] = "train"
    train, rest = train_test_split(main_sample, test_size=C.VAL_FRAC + C.TEST_FRAC,
                                   stratify=main_sample.label, random_state=C.SEED)
    val, test = train_test_split(rest, test_size=C.TEST_FRAC / (C.VAL_FRAC + C.TEST_FRAC),
                                 stratify=rest.label, random_state=C.SEED)
    main_sample.loc[val.index, "split"] = "val"
    main_sample.loc[test.index, "split"] = "test"
    gb_sample["split"] = "galaxy_builder"
    sample = pd.concat([main_sample, gb_sample, moderate], ignore_index=True)
    sample["ra_deg"], sample["dec_deg"] = to_degrees(sample.RA, sample.DEC)

    old = pd.read_csv(C.SAMPLE_CSV) if C.SAMPLE_CSV.exists() else None
    if old is not None:
        sample = sample[~sample.OBJID.isin(old.OBJID)]
        print(f"{len(old)} galaxies already in {C.SAMPLE_CSV}, looking up {len(sample)} new ones")

    batches = [sample.iloc[i:i + BATCH] for i in range(0, len(sample), BATCH)]
    with ThreadPoolExecutor(4) as ex:
        found = pd.concat(tqdm(ex.map(query_sdss, batches), total=len(batches)))
    out = sample.merge(found.drop_duplicates("OBJID"), on="OBJID")
    out = out.merge(gz2, left_on="OBJID", right_on="dr7objid", how="left").drop(columns="dr7objid")

    out = out[(out.petroR90_r > 1) & (out.petroR90_r < 60) & (out.petroR50_r > 0)]
    out["scale"] = (C.FIELD_R90 * out.petroR90_r / C.IMG_SIZE).clip(C.MIN_SCALE, C.MAX_SCALE)
    out["y"] = out.label.map({c: i for i, c in enumerate(C.CLASSES)})
    keep = ["OBJID", "ra_deg", "dec_deg", "label", "y", "split", "NVOTE", "P_EL_DEBIASED", "P_CS_DEBIASED",
            "run", "rerun", "camcol", "field", "petroR50_r", "petroR90_r", "petroMag_r",
            "dered_u", "dered_g", "dered_r", "dered_i", "dered_z", "fracDeV_r", "z", "scale", *GZ2_COLUMNS.values()]
    out = out[keep]
    if old is not None:
        out = pd.concat([old, out], ignore_index=True)
    out.to_csv(C.SAMPLE_CSV, index=False)

    print(f"{len(out)} galaxies written to {C.SAMPLE_CSV}")
    print(out.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("unique SDSS fields:", out[["run", "camcol", "field"]].drop_duplicates().shape[0])


if __name__ == "__main__":
    main()
