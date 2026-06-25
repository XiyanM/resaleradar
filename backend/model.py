from pathlib import Path
import numpy as np
import xgboost as xgb
import shap
from backend.schemas import TOWNS, FLAT_MODELS

MODELS_DIR = Path(__file__).parent.parent / "models"

FEATURE_NAMES = [
    "floor_area_sqm", "lease_commence_date", "year", "transaction_month",
    "storey_midpoint", "remaining_lease_years", "flat_type_encoded",
    "dist_nearest_mrt", "num_mrt_within_1km", "num_mrt_within_2km",
    "dist_nearest_school", "num_schools_within_1km",
    "dist_nearest_primary_school", "num_primary_schools_within_1km",
    "dist_nearest_mall", "num_malls_within_2km",
    "dist_nearest_hawker", "num_hawkers_within_500m",
    "dist_to_cbd", "dist_nearest_expressway", "dist_nearest_bus_stop",
    "num_bus_stops_within_300m", "is_mature_estate",
] + [f"town_{t}" for t in TOWNS] + [f"flat_model_{m}" for m in FLAT_MODELS]


def _load_model(filename: str) -> xgb.Booster:
    model = xgb.Booster()
    model.load_model(MODELS_DIR / filename)
    return model


model_baseline = _load_model("xgb_geo_baseline.json")
model_q10 = _load_model("xgb_geo_q10.json")
model_q50 = _load_model("xgb_geo_q50.json")
model_q90 = _load_model("xgb_geo_q90.json")

explainer = shap.TreeExplainer(model_baseline)


def _expand_categoricals(features: dict) -> dict:
    for town in TOWNS:
        features[f"town_{town}"] = 1 if features["town"] == town else 0
    for model_name in FLAT_MODELS:
        features[f"flat_model_{model_name}"] = 1 if features["flat_model"] == model_name else 0
    del features["town"]
    del features["flat_model"]
    return features


def predict(features: dict) -> dict:
    features = _expand_categoricals(features)
    values = np.array([[features[f] for f in FEATURE_NAMES]], dtype=np.float32)
    dmatrix = xgb.DMatrix(values, feature_names=FEATURE_NAMES)
    q10_pred = float(model_q10.predict(dmatrix)[0])
    q50_pred = float(model_q50.predict(dmatrix)[0])
    q90_pred = float(model_q90.predict(dmatrix)[0])

    lower_bound, predicted_price, upper_bound = sorted([q10_pred, q50_pred, q90_pred])
    shap_vals = explainer.shap_values(values)[0]
    shap_dict = {name: round(float(val), 2) for name, val in zip(FEATURE_NAMES, shap_vals)}

    return {
        "predicted_price": round(predicted_price, 2),
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2),
        "shap_values": shap_dict,
    }