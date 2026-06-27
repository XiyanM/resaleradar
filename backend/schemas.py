from pydantic import BaseModel


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
    floor_area_sqm: float
    lease_commence_date: int
    year: int
    transaction_month: int
    storey_midpoint: float
    remaining_lease_years: float
    flat_type_encoded: int
    dist_nearest_mrt: float
    num_mrt_within_1km: int
    num_mrt_within_2km: int
    dist_nearest_school: float
    num_schools_within_1km: int
    dist_nearest_primary_school: float
    num_primary_schools_within_1km: int
    dist_nearest_mall: float
    num_malls_within_2km: int
    dist_nearest_hawker: float
    num_hawkers_within_500m: int
    dist_to_cbd: float
    dist_nearest_expressway: float
    dist_nearest_bus_stop: float
    num_bus_stops_within_300m: int
    is_mature_estate: int
    town: str
    flat_model: str


class PredictResponse(BaseModel):
    predicted_price: float
    lower_bound: float
    upper_bound: float
    shap_values: dict[str, float]
    quantile_crossing: bool 