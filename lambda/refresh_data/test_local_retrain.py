
import json

from retrain import retrain_and_maybe_promote

BUCKET = "resaleradar"


if __name__ == "__main__":
    result = retrain_and_maybe_promote(
        bucket=BUCKET,
        features_key="processed-data/features_with_geo.csv",
        model_prefix="models-test/",
    )
    print(json.dumps(result, indent=2))