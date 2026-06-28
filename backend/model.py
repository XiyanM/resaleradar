from pathlib import Path
from datetime import date
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

from backend.schemas import TOWNS, FLAT_MODELS

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR   = BASE_DIR / "data" / "processed"

# ── Load models (once at startup) ────────────────────────────────────────────

def _load_model(filename: str) -> xgb.Booster:
    m = xgb.Booster()
    m.load_model(MODELS_DIR / filename)
    return m

model_baseline = _load_model("xgb_geo_baseline.json")
model_q10      = _load_model("xgb_geo_q10.json")
model_q50      = _load_model("xgb_geo_q50.json")
model_q90      = _load_model("xgb_geo_q90.json")

explainer = shap.TreeExplainer(model_baseline)

# ── Load amenity data (once at startup) ──────────────────────────────────────

_mrt      = pd.read_csv(DATA_DIR / "mrt_stations_geocoded.csv", parse_dates=["opening_date"])
_schools  = pd.read_csv(DATA_DIR / "schools_geocoded.csv")
_hawkers  = pd.read_csv(DATA_DIR / "hawker_centres.csv")
_malls    = pd.read_csv(DATA_DIR / "malls_geocoded.csv")
_buses    = pd.read_csv(DATA_DIR / "bus_stops.csv")
_expy     = pd.read_csv(DATA_DIR / "expressway_coords.csv")

# ── Mature estates ────────────────────────────────────────────────────────────

MATURE_ESTATES = {
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT MERAH", "BUKIT TIMAH",
    "CENTRAL AREA", "CLEMENTI", "GEYLANG", "KALLANG/WHAMPOA",
    "MARINE PARADE", "PASIR RIS", "QUEENSTOWN", "SERANGOON",
    "TAO PAYOH", "TOA PAYOH",
}

# ── Flat type encoding (label encoding from nb02) ─────────────────────────────

FLAT_TYPE_MAP = {
    "1 ROOM":           1,
    "2 ROOM":           2,
    "3 ROOM":           3,
    "4 ROOM":           4,
    "5 ROOM":           5,
    "EXECUTIVE":        6,
    "MULTI-GENERATION": 7,
}

# ── CBD coordinates (Raffles Place) ──────────────────────────────────────────

CBD_LAT = 1.2847
CBD_LON = 103.8511

# ── Feature names (must match nb07 training order exactly) ───────────────────

FEATURE_NAMES = (
    [
        "floor_area_sqm", "lease_commence_date", "year", "transaction_month",
        "storey_midpoint", "remaining_lease_years", "flat_type_encoded",
        "dist_nearest_mrt", "num_mrt_within_1km", "num_mrt_within_2km",
        "dist_nearest_school", "num_schools_within_1km",
        "dist_nearest_primary_school", "num_primary_schools_within_1km",
        "dist_nearest_mall", "num_malls_within_2km",
        "dist_nearest_hawker", "num_hawkers_within_500m",
        "dist_to_cbd", "dist_nearest_expressway", "dist_nearest_bus_stop",
        "num_bus_stops_within_300m", "is_mature_estate",
    ]
    + [f"town_{t}"       for t in TOWNS]
    + [f"flat_model_{m}" for m in FLAT_MODELS]
)

# ── Haversine ─────────────────────────────────────────────────────────────────

def _haversine(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """
    Returns distances in km from a single point (lat1, lon1)
    to an array of points (lats, lons).
    All inputs in decimal degrees.
    """
    R = 6371.0
    dlat = np.radians(lats - lat1)
    dlon = np.radians(lons - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lats)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

# ── Geo feature computation ───────────────────────────────────────────────────

def _compute_geo_features(lat: float, lon: float, transaction_year: int, transaction_month: int) -> dict:
    tx_date = date(transaction_year, transaction_month, 1)

    # MRT -- filter to stations open at time of transaction
    mrt_open = _mrt[_mrt["opening_date"].dt.date <= tx_date]
    mrt_dist = _haversine(lat, lon, mrt_open["latitude"].values, mrt_open["longitude"].values)
    dist_nearest_mrt    = float(mrt_dist.min()) if len(mrt_dist) else 99.0
    num_mrt_within_1km  = int((mrt_dist <= 1.0).sum())
    num_mrt_within_2km  = int((mrt_dist <= 2.0).sum())

    # Schools (all schools, no temporal filter)
    school_dist = _haversine(lat, lon, _schools["latitude"].values, _schools["longitude"].values)
    dist_nearest_school         = float(school_dist.min())
    num_schools_within_1km      = int((school_dist <= 1.0).sum())

    # Primary schools only (mainlevel_code == "PRIMARY")
    primary = _schools[_schools["mainlevel_code"] == "PRIMARY"]
    pri_dist = _haversine(lat, lon, primary["latitude"].values, primary["longitude"].values)
    dist_nearest_primary_school    = float(pri_dist.min()) if len(pri_dist) else 99.0
    num_primary_schools_within_1km = int((pri_dist <= 1.0).sum())

    # Hawker centres -- filter to centres open at time of transaction
    hawker_open = _hawkers[
        (_hawkers["status"].str.startswith("Existing")) |
        (_hawkers["completion_year"] <= transaction_year)
    ]
    hawker_dist = _haversine(lat, lon, hawker_open["latitude"].values, hawker_open["longitude"].values)
    dist_nearest_hawker    = float(hawker_dist.min()) if len(hawker_dist) else 99.0
    num_hawkers_within_500m = int((hawker_dist <= 0.5).sum())

    # Malls
    mall_dist = _haversine(lat, lon, _malls["latitude"].values, _malls["longitude"].values)
    dist_nearest_mall   = float(mall_dist.min())
    num_malls_within_2km = int((mall_dist <= 2.0).sum())

    # Bus stops
    bus_dist = _haversine(lat, lon, _buses["latitude"].values, _buses["longitude"].values)
    dist_nearest_bus_stop    = float(bus_dist.min())
    num_bus_stops_within_300m = int((bus_dist <= 0.3).sum())

    # Expressways
    expy_dist = _haversine(lat, lon, _expy["latitude"].values, _expy["longitude"].values)
    dist_nearest_expressway = float(expy_dist.min())

    # CBD
    dist_to_cbd = float(_haversine(lat, lon, np.array([CBD_LAT]), np.array([CBD_LON]))[0])

    return {
        "dist_nearest_mrt":             dist_nearest_mrt,
        "num_mrt_within_1km":           num_mrt_within_1km,
        "num_mrt_within_2km":           num_mrt_within_2km,
        "dist_nearest_school":          dist_nearest_school,
        "num_schools_within_1km":       num_schools_within_1km,
        "dist_nearest_primary_school":  dist_nearest_primary_school,
        "num_primary_schools_within_1km": num_primary_schools_within_1km,
        "dist_nearest_mall":            dist_nearest_mall,
        "num_malls_within_2km":         num_malls_within_2km,
        "dist_nearest_hawker":          dist_nearest_hawker,
        "num_hawkers_within_500m":      num_hawkers_within_500m,
        "dist_nearest_bus_stop":        dist_nearest_bus_stop,
        "num_bus_stops_within_300m":    num_bus_stops_within_300m,
        "dist_nearest_expressway":      dist_nearest_expressway,
        "dist_to_cbd":                  dist_to_cbd,
    }

# ── Categorical expansion ─────────────────────────────────────────────────────

def _expand_categoricals(features: dict) -> dict:
    for town in TOWNS:
        features[f"town_{town}"] = 1 if features["town"] == town else 0
    for model_name in FLAT_MODELS:
        features[f"flat_model_{model_name}"] = 1 if features["flat_model"] == model_name else 0
    del features["town"]
    del features["flat_model"]
    return features

# ── Main predict function ─────────────────────────────────────────────────────

def predict(req: dict) -> dict:
    # Geo features
    geo = _compute_geo_features(req["lat"], req["lon"], req["transaction_year"], req["transaction_month"])

    # Apply what-if overrides if present
    overrides = req.get("feature_overrides") or {}
    geo.update(overrides)

    # Derived scalar features
    remaining_lease_years = req.get("remaining_lease_years") or (99 - (req["transaction_year"] - req["lease_commence_date"]))
    flat_type_encoded     = FLAT_TYPE_MAP[req["flat_type"]]
    is_mature_estate      = 1 if req["town"] in MATURE_ESTATES else 0

    # Assemble feature dict
    features = {
        "floor_area_sqm":        req["floor_area_sqm"],
        "lease_commence_date":   req["lease_commence_date"],
        "year":                  req["transaction_year"],
        "transaction_month":     req["transaction_month"],
        "storey_midpoint":       req["storey_midpoint"],
        "remaining_lease_years": remaining_lease_years,
        "flat_type_encoded":     flat_type_encoded,
        "is_mature_estate":      is_mature_estate,
        "town":                  req["town"],
        "flat_model":            req["flat_model"],
        **geo,
    }

    # Snapshot feature values before expanding categoricals (for frontend slider init)
    feature_values = {
        "dist_nearest_mrt": geo["dist_nearest_mrt"],
        "dist_to_cbd":      geo["dist_to_cbd"],
    }

    features = _expand_categoricals(features)

    # Build numpy array in exact training order
    values  = np.array([[features[f] for f in FEATURE_NAMES]], dtype=np.float32)
    dmatrix = xgb.DMatrix(values, feature_names=FEATURE_NAMES)

    # Quantile predictions
    q10 = float(model_q10.predict(dmatrix)[0])
    q50 = float(model_q50.predict(dmatrix)[0])
    q90 = float(model_q90.predict(dmatrix)[0])

    crossing = not (q10 <= q50 <= q90)
    lower_bound, predicted_price, upper_bound = sorted([q10, q50, q90])

    # SHAP (baseline model only)
    shap_vals = explainer.shap_values(values)[0]
    shap_dict = {name: round(float(val), 2) for name, val in zip(FEATURE_NAMES, shap_vals)}

    return {
        "predicted_price":   round(predicted_price, 2),
        "lower_bound":       round(lower_bound, 2),
        "upper_bound":       round(upper_bound, 2),
        "shap_values":       shap_dict,
        "quantile_crossing": crossing,
        "feature_values":    feature_values,
    }