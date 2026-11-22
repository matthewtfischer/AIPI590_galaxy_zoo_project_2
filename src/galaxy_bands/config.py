from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
GZ1_TABLE = RAW / "GalaxyZoo1_DR_table2.csv"
GZ2_TABLE = RAW / "gz2_hart16.csv.gz"
GALAXY_BUILDER = RAW / "galaxy-builder-catalog.csv"
SAMPLE_CSV = DATA / "sample.csv"
CUTOUT_DIR = DATA / "cutouts"
CUTOUT_STATUS = DATA / "cutout_status.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS = ROOT / "models"

SEED = 590

BANDS = ["u", "g", "r", "i", "z"]
CLASSES = ["spiral", "elliptical"]

# sample selection
MIN_VOTES = 30
CONF_THRESHOLD = 0.85
N_PER_CLASS = 3000
# extra test set: galaxies where 70-85% of the debiased vote went to the winning type
MODERATE_RANGE = (0.70, 0.85)
N_MODERATE_PER_CLASS = 500

# cutouts are 4 * petroR90 across, so R90 is 32 px from the center
IMG_SIZE = 128
FIELD_R90 = 4.0
R90_PIX = IMG_SIZE / FIELD_R90
MIN_SCALE, MAX_SCALE = 0.05, 1.5   # arcsec/pixel

SDSS_SQL_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"
SDSS_FRAME_URL = ("https://data.sdss.org/sas/dr17/eboss/photoObj/frames/{rerun}/{run}/{camcol}/"
                  "frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2")

VAL_FRAC = 0.15
TEST_FRAC = 0.15

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass
