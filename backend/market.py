from pathlib import Path
from functools import lru_cache
import pandas as pd

BASE_DIR  = Path(__file__).parent.parent
DATA_DIR  = BASE_DIR / "data"

def _toTitleCase(s: str) -> str:
    return s.lower().replace('/', '/ ').title().replace('/ ', '/')

@lru_cache(maxsize=1)
def get_market_data() -> dict:
    # ── Resale transactions ───────────────────────────────────────────────
    df = pd.read_csv(DATA_DIR / "raw" / "hdb_resale_transactions.csv")
    df["month"] = pd.to_datetime(df["month"])
    df["year_month"] = df["month"].dt.to_period("M").astype(str)

    # Town medians -- all time, nominal prices
    town_medians_raw = (
        df.groupby("town")["resale_price"]
        .median()
        .round(-2)
        .astype(int)
        .sort_values(ascending=False)
        .to_dict()
    )
    # Convert keys to title case for display
    town_medians = {_toTitleCase(k): int(v) for k, v in town_medians_raw.items()}

    # Monthly median trend -- all towns combined, nominal prices
    monthly = (
        df.groupby("year_month")["resale_price"]
        .median()
        .round(-2)
        .astype(int)
    )
    trend_labels = list(monthly.index)
    trend_values = [int(v) for v in monthly.values]

    # ── CPI ───────────────────────────────────────────────────────────────
    cpi_df = pd.read_csv(DATA_DIR / "raw" / "cpi.csv")

    # Find the "All Items" row
    all_items = cpi_df[cpi_df.iloc[:, 0].str.strip() == "All Items"]
    if all_items.empty:
        current_cpi = 102.052  # fallback to base year
    else:
        row = all_items.iloc[0, 1:]  # skip the label column
        # Drop 'na' strings and get the last valid value
        valid = row[row.apply(lambda x: str(x).strip().lower() != "na")]
        current_cpi = float(valid.iloc[0])
    return {
        "town_medians":  town_medians,
        "trend_labels":  trend_labels,
        "trend_values":  trend_values,
        "current_cpi":   current_cpi,
        "base_cpi":      100.662,
    }