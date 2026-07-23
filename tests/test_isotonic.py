import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from scripts.train import build_features

df = pd.read_csv("data/processed/features_with_geo.csv")
df_features = build_features(df)
feat_cols = [c for c in df_features.columns if c != "resale_price_real"]
lease_idx = feat_cols.index("remaining_lease_years")

model = xgb.Booster()
model.load_model("models_constrained_v2/xgb_geo_q50.json")  # same model, no retraining needed

sample = df_features[feat_cols].sample(n=2000, random_state=44).reset_index(drop=True)
lease_range = list(range(40, 100, 2))
full_grid = list(range(1, 100))  # correction sweep uses the full valid range

violations = 0
violation_sizes = []

for i in range(len(sample)):
    base_row = sample.iloc[i].values.astype(np.float32)

    # Full sweep for isotonic fit
    rows = np.tile(base_row, (len(full_grid), 1))
    rows[:, lease_idx] = full_grid
    dmat = xgb.DMatrix(rows, feature_names=feat_cols)
    raw_preds = model.predict(dmat)

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    corrected = iso.fit_transform(full_grid, raw_preds)

    # Read off corrected values at our usual test points (40..98 step 2)
    test_preds = np.interp(lease_range, full_grid, corrected)
    diffs = np.diff(test_preds)
    if (diffs < 0).any():
        violations += 1
        violation_sizes.append(-diffs.min())

print(f"Rows tested: {len(sample)}")
print(f"Rows with >=1 violation: {violations} ({violations/len(sample)*100:.1f}%)")
if violation_sizes:
    print(f"Median violation size: SGD {np.median(violation_sizes):,.0f}")
    print(f"Largest violation:     SGD {max(violation_sizes):,.0f}")