import io
from pathlib import Path
from functools import lru_cache
import boto3
import pandas as pd
import logging

logger = logging.getLogger(__name__)

S3_BUCKET = "resaleradar"
RAW_PREFIX = "raw-data/"

s3 = boto3.client("s3")


def _read_csv_from_s3(key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def _toTitleCase(s: str) -> str:
    return s.lower().replace('/', '/ ').title().replace('/ ', '/')


def _compute_base_cpi(cpi_df: pd.DataFrame) -> float:
    all_items = cpi_df[cpi_df.iloc[:, 0].str.strip() == "All Items"].iloc[0]
    cpi_long = all_items.iloc[1:].reset_index()
    cpi_long.columns = ["month_str", "cpi"]
    cpi_long = cpi_long[cpi_long["cpi"].astype(str).str.strip().str.lower() != "na"].copy()
    cpi_long["cpi"] = cpi_long["cpi"].astype(float)
    cpi_long["transaction_date"] = pd.to_datetime(cpi_long["month_str"], format="%Y%b")

    base = cpi_long[
        (cpi_long["transaction_date"] >= "2024-01-01") &
        (cpi_long["transaction_date"] < "2025-01-01")
    ]["cpi"].mean()
    return float(base)


@lru_cache(maxsize=1)
def get_market_data() -> dict:
    # ── Resale transactions (from S3, refreshed monthly by the Lambda) ─────
    df = _read_csv_from_s3(f"{RAW_PREFIX}hdb_resale_transactions.csv")
    df["month"] = pd.to_datetime(df["month"])
    df["year_month"] = df["month"].dt.to_period("M").astype(str)

    town_medians_raw = (
        df.groupby("town")["resale_price"]
        .median()
        .round(-2)
        .astype(int)
        .sort_values(ascending=False)
        .to_dict()
    )
    town_medians = {_toTitleCase(k): int(v) for k, v in town_medians_raw.items()}

    monthly = (
        df.groupby("year_month")["resale_price"]
        .median()
        .round(-2)
        .astype(int)
    )
    trend_labels = list(monthly.index)
    trend_values = [int(v) for v in monthly.values]

    # ── CPI (from S3, refreshed monthly by the Lambda) ──────────────────────
    cpi_df = _read_csv_from_s3(f"{RAW_PREFIX}cpi.csv")

    all_items = cpi_df[cpi_df.iloc[:, 0].str.strip() == "All Items"]
    base_cpi = _compute_base_cpi(cpi_df)

    if all_items.empty:
        logger.warning("CPI lookup failed: 'All Items' row not found in cpi.csv. Falling back to base_cpi (no adjustment applied).")
        current_cpi = base_cpi
    else:
        row = all_items.iloc[0, 1:]
        valid = row[row.apply(lambda x: str(x).strip().lower() != "na")]
        if valid.empty:
            logger.warning("CPI lookup failed: 'All Items' row found but every value is 'na'. Falling back to base_cpi.")
            current_cpi = base_cpi
        else:
            current_cpi = float(valid.iloc[0])

    return {
        "town_medians":  town_medians,
        "trend_labels":  trend_labels,
        "trend_values":  trend_values,
        "current_cpi":   current_cpi,
        "base_cpi":      base_cpi,
    }