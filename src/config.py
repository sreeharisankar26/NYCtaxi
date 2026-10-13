"""Paths and fixed parameters shared by every pipeline step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SQL_DIR = ROOT / "sql"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
TABLE_DIR = REPORTS_DIR / "tables"
DB_PATH = PROCESSED_DIR / "taxi.duckdb"

YEAR = 2024
RAW_GLOB = RAW_DIR / f"yellow_tripdata_{YEAR}-*.parquet"
ZONE_CSV = RAW_DIR / "taxi_zone_lookup.csv"

SEED = 20240101

# Cleaning thresholds (see sql/01_clean.sql)
MIN_DURATION_S = 60          # under 1 minute
MAX_DURATION_S = 3 * 3600    # over 3 hours
MAX_SPEED_MPH = 80

# A zone is "active" (enough data for per-zone inference) if it averages
# at least this many pickups per day over the year.
ACTIVE_ZONE_MIN_DAILY_MEAN = 10

# Hour blocks for the logit model (inclusive hour ranges)
HOUR_BLOCKS = {
    "night_00_05": range(0, 6),
    "am_peak_06_09": range(6, 10),
    "midday_10_15": range(10, 16),
    "pm_peak_16_19": range(16, 20),
    "evening_20_23": range(20, 24),
}

# Forecast validation
N_FOLDS = 12
FOLD_STEP_DAYS = 19          # not a multiple of 7, so fold origins rotate through weekdays
FIRST_ORIGIN = "2024-04-01 00:00"
HORIZONS = (24, 168)

ALPHA = 0.05


def ensure_dirs() -> None:
    for d in (PROCESSED_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
