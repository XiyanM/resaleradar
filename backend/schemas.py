from pydantic import BaseModel
from typing import Optional


TOWNS = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH",
    "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
    "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
    "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL",
    "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
    "TOA PAYOH", "WOODLANDS", "YISHUN",
]

FLAT_MODELS = [
    "2-room", "3Gen", "Adjoined flat", "Apartment", "DBSS", "Improved",
    "Improved-Maisonette", "Maisonette", "Model A", "Model A-Maisonette",
    "Model A2", "Multi Generation", "New Generation", "Premium Apartment",
    "Premium Apartment Loft", "Premium Maisonette", "Simplified", "Standard",
    "Terrace", "Type S1", "Type S2",
]


class PredictRequest(BaseModel):
    lat: float
    lon: float
    town: str
    flat_type: str
    flat_model: str
    floor_area_sqm: float
    storey_midpoint: float
    lease_commence_date: int
    transaction_year: int
    transaction_month: int
    remaining_lease_years: Optional[float] = None
    feature_overrides: Optional[dict[str, float]] = None

class PredictResponse(BaseModel):
    predicted_price: float
    lower_bound: float
    upper_bound: float
    shap_values: dict[str, float]
    quantile_crossing: bool
    feature_values: dict[str, float]