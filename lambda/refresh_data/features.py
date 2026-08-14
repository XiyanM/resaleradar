"""
Feature engineering for the ResaleRadar Lambda, ported from nb04 + nb06.

Regenerates processed-data/features_with_geo.csv in S3 from whatever is
currently in raw-data/ (refreshed monthly by lambda_function.py) plus the
static reference data uploaded once to reference-data/.

"""
import io
import logging
from math import radians

import boto3
import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

CBD_LAT, CBD_LON = 1.2830, 103.8513

MATURE_ESTATES = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH",
    "CENTRAL AREA", "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA",
    "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON",
    "TAMPINES", "TOA PAYOH",
]

BATCH_SIZE = 10_000  # transactions per chunk, bounds peak matrix memory


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def _write_csv_to_s3(df: pd.DataFrame, bucket: str, key: str) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    logger.info("Wrote %d rows to s3://%s/%s", len(df), bucket, key)


# ── Vectorized Haversine (chunked) ───────────────────────────────────────────

def _haversine_matrix(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """
    Great-circle distance in km between every point in (lat1, lon1) and every
    point in (lat2, lon2), as a (len(lat1), len(lat2)) matrix. Same formula
    as nb06's haversine(), broadcast across all pairs at once.
    """
    lat1r = np.radians(lat1)[:, None]
    lon1r = np.radians(lon1)[:, None]
    lat2r = np.radians(lat2)[None, :]
    lon2r = np.radians(lon2)[None, :]
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1 - a, 0, None)))
    return 6371.0 * c


def _batched_distance_features(
    lats: np.ndarray, lons: np.ndarray, amenity_df: pd.DataFrame,
    thresholds: list, transaction_dates: pd.Series = None,
    date_col: str = None, allow_null_date: bool = False,
) -> dict:
    """
    Nearest distance + counts-within-radius to amenity_df, for every
    transaction, computed in memory-safe batches.

    If date_col is given, an amenity only counts for a transaction if its
    date_col <= that transaction's date (temporal leakage prevention, same
    as nb06's MRT opening-date and hawker completion-date filtering).
    allow_null_date=True also counts amenities with a missing date_col
    value (matches nb06's hawker handling, where 2 existing hawkers had no
    completion date on record and were always included).
    """
    n = len(lats)
    amenity_lat = amenity_df["latitude"].values
    amenity_lon = amenity_df["longitude"].values

    if date_col:
        amenity_dates = amenity_df[date_col].values
        tdates = transaction_dates.values

    dist_nearest = np.empty(n)
    counts = {t: np.empty(n, dtype=int) for t in thresholds}

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        dist_matrix = _haversine_matrix(lats[start:end], lons[start:end], amenity_lat, amenity_lon)

        if date_col:
            available = tdates[start:end][:, None] >= amenity_dates[None, :]
            if allow_null_date:
                available = available | pd.isna(amenity_dates)[None, :]
            dist_matrix = np.where(available, dist_matrix, np.inf)
            count_mask_base = available
        else:
            count_mask_base = np.ones_like(dist_matrix, dtype=bool)

        row_min = dist_matrix.min(axis=1)
        row_min[np.isinf(row_min)] = np.nan  # no amenity available yet for this row
        dist_nearest[start:end] = row_min

        for t in thresholds:
            counts[t][start:end] = ((dist_matrix < t) & count_mask_base).sum(axis=1)

    result = {"dist_nearest": dist_nearest}
    result.update({f"count_within_{t}": counts[t] for t in thresholds})
    return result


# ── CPI adjustment (identical logic to nb06) ─────────────────────────────────

def _apply_cpi_adjustment(df: pd.DataFrame, cpi_df: pd.DataFrame) -> pd.DataFrame:
    cpi_all = cpi_df[cpi_df["DataSeries"] == "All Items"].iloc[0]
    cpi_long = cpi_all.drop("DataSeries").reset_index()
    cpi_long.columns = ["month_str", "cpi"]
    cpi_long = cpi_long[cpi_long["cpi"] != "na"].copy()
    cpi_long["cpi"] = cpi_long["cpi"].astype(float)
    cpi_long["transaction_date"] = pd.to_datetime(cpi_long["month_str"], format="%Y%b")
    cpi_long = cpi_long[cpi_long["transaction_date"] >= "2017-01-01"].sort_values("transaction_date")

    base_cpi = cpi_long[
        (cpi_long["transaction_date"] >= "2024-01-01") &
        (cpi_long["transaction_date"] < "2025-01-01")
    ]["cpi"].mean()    
    logger.info("Base CPI (2024 average): %.3f", base_cpi)

    df = df.merge(cpi_long[["transaction_date", "cpi"]], on="transaction_date", how="left")
    missing = df["cpi"].isna().sum()
    if missing:
        latest_cpi = cpi_long["cpi"].iloc[-1]
        logger.info("Filling %d rows with latest available CPI: %.3f", missing, latest_cpi)
        df["cpi"] = df["cpi"].fillna(latest_cpi)

    df["resale_price_real"] = df["resale_price"] * (base_cpi / df["cpi"])
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def rebuild_features_with_geo(bucket: str) -> dict:
    """
    Regenerate processed-data/features_with_geo.csv from the current
    raw-data/ transactions + CPI and the static reference-data/ amenities.
    Returns a summary dict for logging.
    """
    df = _read_csv_from_s3(bucket, "raw-data/hdb_resale_transactions.csv")
    cpi_df = _read_csv_from_s3(bucket, "raw-data/cpi.csv")
    geocoded = _read_csv_from_s3(bucket, "reference-data/geocoded_addresses.csv")

    mrt = _read_csv_from_s3(bucket, "reference-data/mrt_stations_geocoded.csv")
    mrt["opening_date"] = pd.to_datetime(mrt["opening_date"])
    schools = _read_csv_from_s3(bucket, "reference-data/schools_geocoded.csv")
    hawkers = _read_csv_from_s3(bucket, "reference-data/hawker_centres.csv")
    # hawker_centres.csv has completion_year (an integer), not a full date --
    # est_completion_date was cleaned into this integer form in nb05, before
    # this file was saved. The 2 rows with a missing completion_year are
    # both status "Existing" (per project notes), which is exactly why
    # treating a missing value as "always available" (same as nb06's isna()
    # handling) is still correct here, just compared by year instead of date.
    malls = _read_csv_from_s3(bucket, "reference-data/malls_geocoded.csv")
    expressways = _read_csv_from_s3(bucket, "reference-data/expressway_coords.csv")
    bus_stops = _read_csv_from_s3(bucket, "reference-data/bus_stops.csv")
    primary_schools = schools[schools["mainlevel_code"].isin(["PRIMARY", "MIXED LEVEL (P1-S4)"])].copy()

    logger.info("Loaded %d transactions, %d geocoded addresses", len(df), len(geocoded))

    # Merge coordinates; detect and drop addresses not in the reference lookup
    df = df.merge(geocoded, on=["block", "street_name"], how="left")
    unmatched = df["latitude"].isna().sum()
    if unmatched:
        logger.warning(
            "%d transactions (%.2f%%) have addresses not in geocoded_addresses.csv "
            "-- dropping from output. See module docstring for why these aren't "
            "auto-geocoded.",
            unmatched, 100 * unmatched / len(df),
        )
        df = df[df["latitude"].notna()].reset_index(drop=True)

    df["transaction_date"] = pd.to_datetime(df["month"])
    df["transaction_year"] = df["transaction_date"].dt.year
    lats = df["latitude"].values
    lons = df["longitude"].values

    mrt_feats = _batched_distance_features(
        lats, lons, mrt, thresholds=[1, 2],
        transaction_dates=df["transaction_date"], date_col="opening_date",
    )
    school_feats = _batched_distance_features(lats, lons, schools, thresholds=[1])
    primary_feats = _batched_distance_features(lats, lons, primary_schools, thresholds=[1])
    mall_feats = _batched_distance_features(lats, lons, malls, thresholds=[2])
    hawker_feats = _batched_distance_features(
        lats, lons, hawkers, thresholds=[0.5],
        transaction_dates=df["transaction_year"], date_col="completion_year",
        allow_null_date=True,
    )
    expressway_feats = _batched_distance_features(lats, lons, expressways, thresholds=[])
    bus_feats = _batched_distance_features(lats, lons, bus_stops, thresholds=[0.3])

    df["dist_nearest_mrt"] = mrt_feats["dist_nearest"]
    df["num_mrt_within_1km"] = mrt_feats["count_within_1"]
    df["num_mrt_within_2km"] = mrt_feats["count_within_2"]
    df["dist_nearest_school"] = school_feats["dist_nearest"]
    df["num_schools_within_1km"] = school_feats["count_within_1"]
    df["dist_nearest_primary_school"] = primary_feats["dist_nearest"]
    df["num_primary_schools_within_1km"] = primary_feats["count_within_1"]
    df["dist_nearest_mall"] = mall_feats["dist_nearest"]
    df["num_malls_within_2km"] = mall_feats["count_within_2"]
    df["dist_nearest_hawker"] = hawker_feats["dist_nearest"]
    df["num_hawkers_within_500m"] = hawker_feats["count_within_0.5"]
    df["dist_to_cbd"] = _haversine_matrix(lats, lons, np.array([CBD_LAT]), np.array([CBD_LON]))[:, 0]
    df["dist_nearest_expressway"] = expressway_feats["dist_nearest"]
    df["dist_nearest_bus_stop"] = bus_feats["dist_nearest"]
    df["num_bus_stops_within_300m"] = bus_feats["count_within_0.3"]

    df["is_mature_estate"] = df["town"].isin(MATURE_ESTATES).astype(int)

    df = _apply_cpi_adjustment(df, cpi_df)

    _write_csv_to_s3(df, bucket, "processed-data/features_with_geo.csv")

    return {
        "rows_written": len(df),
        "rows_dropped_unmatched_address": int(unmatched),
        "columns": df.shape[1],
    }