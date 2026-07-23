import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
from scripts.train import build_features

df = pd.read_csv("data/processed/features_with_geo.csv")
df_features = build_features(df)

X = df_features.drop(columns=["resale_price_real"])
y = df_features["resale_price_real"]

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

model = xgb.XGBRegressor()
model.load_model("models_constrained_v2/xgb_geo_baseline.json")

pred_test = model.predict(X_test)
print(f"Test R²:   {r2_score(y_test, pred_test):.4f}")
print(f"Test RMSE: {root_mean_squared_error(y_test, pred_test):,.2f}")