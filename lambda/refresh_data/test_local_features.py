"""
Local smoke test for features.py -- rebuilds features_with_geo.csv for real
against live S3 data (raw-data/ + reference-data/) and overwrites
processed-data/features_with_geo.csv with the freshly computed version.

Needs your local AWS credentials and: boto3, pandas, numpy.

Usage:
    python test_local_features.py
"""

import json

from features import rebuild_features_with_geo

BUCKET = "resaleradar"

if __name__ == "__main__":
    result = rebuild_features_with_geo(BUCKET)
    print(json.dumps(result, indent=2))
