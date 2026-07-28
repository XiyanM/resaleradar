
import io
import json
import logging
from datetime import datetime, timezone

import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")

MODEL_NAMES = {
    "baseline": "xgb_geo_baseline.json",
    0.1: "xgb_geo_q10.json",
    0.5: "xgb_geo_q50.json",
    0.9: "xgb_geo_q90.json",
}


# ── Feature engineering (identical to train.py) ──────────────────────────────

def extract_storey_midpoint(storey_range: str) -> float:
    low, high = storey_range.split(" TO ")
    return (int(low) + int(high)) / 2


def parse_remaining_lease(lease_str: str) -> float:
    parts = lease_str.split(" ")
    years = int(parts[0])
    months = int(parts[2]) if "month" in lease_str else 0
    return years + (months / 12)


FLAT_TYPE_ORDER = {
    "1 ROOM": 1, "2 ROOM": 2, "3 ROOM": 3,
    "4 ROOM": 4, "5 ROOM": 5, "EXECUTIVE": 6, "MULTI-GENERATION": 7,
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["storey_midpoint"] = df["storey_range"].apply(extract_storey_midpoint)
    df["remaining_lease_years"] = df["remaining_lease"].apply(parse_remaining_lease)
    df["flat_type_encoded"] = df["flat_type"].map(FLAT_TYPE_ORDER)

    town_dummies = pd.get_dummies(df["town"], prefix="town")
    flat_model_dummies = pd.get_dummies(df["flat_model"], prefix="flat_model")

    df_features = pd.DataFrame({
        "floor_area_sqm": df["floor_area_sqm"],
        "year": df["month"].str[:4].astype(int),
        "transaction_month": df["month"].str[5:7].astype(int),
        "storey_midpoint": df["storey_midpoint"],
        "remaining_lease_years": df["remaining_lease_years"],
        "flat_type_encoded": df["flat_type_encoded"],
        "dist_nearest_mrt": df["dist_nearest_mrt"],
        "num_mrt_within_1km": df["num_mrt_within_1km"],
        "num_mrt_within_2km": df["num_mrt_within_2km"],
        "dist_nearest_school": df["dist_nearest_school"],
        "num_schools_within_1km": df["num_schools_within_1km"],
        "dist_nearest_primary_school": df["dist_nearest_primary_school"],
        "num_primary_schools_within_1km": df["num_primary_schools_within_1km"],
        "dist_nearest_mall": df["dist_nearest_mall"],
        "num_malls_within_2km": df["num_malls_within_2km"],
        "dist_nearest_hawker": df["dist_nearest_hawker"],
        "num_hawkers_within_500m": df["num_hawkers_within_500m"],
        "dist_to_cbd": df["dist_to_cbd"],
        "dist_nearest_expressway": df["dist_nearest_expressway"],
        "dist_nearest_bus_stop": df["dist_nearest_bus_stop"],
        "num_bus_stops_within_300m": df["num_bus_stops_within_300m"],
        "is_mature_estate": df["is_mature_estate"],
    })

    df_features = pd.concat([df_features, town_dummies, flat_model_dummies], axis=1)
    df_features["resale_price_real"] = df["resale_price_real"]
    return df_features


# ── S3 helpers ────────────────────────────────────────────────────────────────

def _read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def _read_json_from_s3(bucket: str, key: str):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except s3.exceptions.NoSuchKey:
        return None


def _write_json_to_s3(bucket: str, key: str, data: dict) -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(data, indent=2))


def _write_model_to_s3(model: xgb.XGBRegressor, bucket: str, key: str) -> None:
    # write to Lambda's writable /tmp first, then upload that file to S3.
    tmp_path = f"/tmp/{key.split('/')[-1]}"
    model.save_model(tmp_path)
    s3.upload_file(tmp_path, bucket, key)


# ── Training + evaluation ─────────────────────────────────────────────────────

def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def train_candidate(bucket: str, features_key: str):
    """Train baseline + 3 quantile models, return them plus test-set metrics."""
    df = _read_csv_from_s3(bucket, features_key)
    df_features = build_features(df)

    X = df_features.drop(columns=["resale_price_real"])
    y = df_features["resale_price_real"]

    MONOTONIC_INCREASING = {"remaining_lease_years"}
    monotone_constraints = tuple(1 if col in MONOTONIC_INCREASING else 0 for col in X.columns)
    monotone_str = "(" + ",".join(str(v) for v in monotone_constraints) + ")"

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=44)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=44)

    logger.info("Train: %d rows | Val: %d rows | Test: %d rows | Features: %d",
                len(X_train), len(X_val), len(X_test), X_train.shape[1])

    model_baseline = xgb.XGBRegressor(
        n_estimators=5000, learning_rate=0.1, max_depth=6, random_state=44,
        eval_metric="rmse", early_stopping_rounds=50, tree_method="hist",
        monotone_constraints=monotone_str,
    )
    model_baseline.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    quantile_models = {}
    for q in [0.1, 0.5, 0.9]:
        logger.info("Training quantile %.1f model...", q)
        model_q = xgb.XGBRegressor(
            n_estimators=2000, learning_rate=0.1, max_depth=6,
            objective="reg:quantileerror", quantile_alpha=q, tree_method="hist",
            random_state=44, early_stopping_rounds=50,
            monotone_constraints=monotone_str, max_bin=512,
        )
        model_q.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        quantile_models[q] = model_q
        logger.info("  Done. Best iteration: %s", model_q.best_iteration)

    
    y_pred_test = quantile_models[0.5].predict(X_test)
    test_mape = _mape(y_test.values, y_pred_test)
    test_rmse = float(np.sqrt(np.mean((y_test.values - y_pred_test) ** 2)))

    y_pred_baseline = model_baseline.predict(X_test)
    baseline_mape = _mape(y_test.values, y_pred_baseline)
    baseline_rmse = float(np.sqrt(np.mean((y_test.values - y_pred_baseline) ** 2)))

    logger.info("Candidate q50 test MAPE: %.2f%% | RMSE: %.2f", test_mape, test_rmse)
    logger.info("Candidate baseline test MAPE: %.2f%% | RMSE: %.2f", baseline_mape, baseline_rmse)

    models = {"baseline": model_baseline, **quantile_models}
    metrics = {
        "mape": test_mape,
        "rmse": test_rmse,
        "baseline_mape": baseline_mape,
        "baseline_rmse": baseline_rmse,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_count": X_train.shape[1],
    }
    return models, metrics


def retrain_and_maybe_promote(bucket: str, features_key: str = "processed-data/features_with_geo.csv",
                               model_prefix: str = "models/") -> dict:
    """
    Train a candidate model, compare it against the current live model's
    recorded metrics, and only overwrite the live models if the candidate
    is actually better. Returns a summary dict for logging.
    """
    metrics_key = f"{model_prefix}metrics.json"
    champion_metrics = _read_json_from_s3(bucket, metrics_key)

    candidate_models, candidate_metrics = train_candidate(bucket, features_key)

    if champion_metrics is None:
        # No champion exists yet -- this is the first run, promote unconditionally.
        promote = True
        reason = "no existing champion; promoting first trained model"
    elif candidate_metrics["mape"] < champion_metrics["mape"]:
        promote = True
        reason = f"candidate MAPE {candidate_metrics['mape']:.2f}% beat champion {champion_metrics['mape']:.2f}%"
    else:
        promote = False
        reason = f"candidate MAPE {candidate_metrics['mape']:.2f}% did not beat champion {champion_metrics['mape']:.2f}%"

    logger.info("Promotion decision: %s (%s)", promote, reason)

    if promote:
        for name, model in candidate_models.items():
            _write_model_to_s3(model, bucket, f"{model_prefix}{MODEL_NAMES[name]}")
        _write_json_to_s3(bucket, metrics_key, candidate_metrics)

    return {
        "promoted": promote,
        "reason": reason,
        "candidate_metrics": candidate_metrics,
        "champion_metrics": champion_metrics,
    }