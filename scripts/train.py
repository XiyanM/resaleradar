

import argparse
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


# ── Feature engineering (identical to nb07) ──────────────────────────────────

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
        "lease_commence_date": df["lease_commence_date"],
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


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_save(args):
    input_path = os.path.join(args.train, "features_with_geo.csv")
    df = pd.read_csv(input_path)
    df_features = build_features(df)

    X = df_features.drop(columns=["resale_price_real"])
    y = df_features["resale_price_real"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=44
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=44
    )

    print(f"Train: {X_train.shape[0]:,} rows | Val: {X_val.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")
    print(f"Feature columns: {X_train.shape[1]}")

    # Baseline model (squared error), used exclusively for SHAP downstream
    model_baseline = xgb.XGBRegressor(
        n_estimators=5000,
        learning_rate=0.1,
        max_depth=6,
        random_state=44,
        eval_metric="rmse",
        early_stopping_rounds=50,
        tree_method="hist",
    )
    model_baseline.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    # Quantile models
    quantile_models = {}
    for q in [0.1, 0.5, 0.9]:
        print(f"Training quantile {q} model...")
        model_q = xgb.XGBRegressor(
            n_estimators=2000,
            learning_rate=0.1,
            max_depth=6,
            objective="reg:quantileerror",
            quantile_alpha=q,
            tree_method="hist",
            random_state=44,
            early_stopping_rounds=50,
        )
        model_q.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        quantile_models[q] = model_q
        print(f"  Done. Best iteration: {model_q.best_iteration}")

    # Save all four models to SM_MODEL_DIR 
    os.makedirs(args.model_dir, exist_ok=True)
    model_baseline.save_model(os.path.join(args.model_dir, "xgb_geo_baseline.json"))
    quantile_models[0.1].save_model(os.path.join(args.model_dir, "xgb_geo_q10.json"))
    quantile_models[0.5].save_model(os.path.join(args.model_dir, "xgb_geo_q50.json"))
    quantile_models[0.9].save_model(os.path.join(args.model_dir, "xgb_geo_q90.json"))
    print(f"Models saved to {args.model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()

    train_and_save(args)