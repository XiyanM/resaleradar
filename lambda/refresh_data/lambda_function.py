import json
import logging
import os
import time
import urllib.request

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CPI_DATASET_ID = "d_bdaff844e3ef89d39fceb962ff8f0791"
TRANSACTIONS_DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"

API_HOST = "https://api-open.data.gov.sg"

S3_BUCKET = os.environ["S3_BUCKET"]
S3_PREFIX = os.environ.get("S3_PREFIX", "raw-data/")

s3 = boto3.client("s3")


def _api_get(path: str) -> dict:
    req = urllib.request.Request(API_HOST + path, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.loads(response.read())


def _download_dataset_csv(dataset_id: str, max_polls: int = 10, poll_delay_seconds: int = 3) -> bytes:
    _api_get(f"/v1/public/api/datasets/{dataset_id}/initiate-download")

    download_url = None
    for attempt in range(max_polls):
        poll_response = _api_get(f"/v1/public/api/datasets/{dataset_id}/poll-download")
        data = poll_response.get("data", {})
        status = data.get("status")
        logger.info("Poll %s for %s: status=%s", attempt + 1, dataset_id, status)

        if data.get("url"):
            download_url = data["url"]
            break

        time.sleep(poll_delay_seconds)

    if download_url is None:
        raise TimeoutError(f"Dataset {dataset_id} did not become ready after {max_polls} polls")


    with urllib.request.urlopen(download_url, timeout=60) as csv_response:
        return csv_response.read()


def _upload_to_s3(csv_bytes: bytes, filename: str) -> None:
    key = f"{S3_PREFIX}{filename}"
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=csv_bytes)
    logger.info("Uploaded %d bytes to s3://%s/%s", len(csv_bytes), S3_BUCKET, key)


def handler(event, context):
    results = {}

    for label, dataset_id, filename in [
        ("transactions", TRANSACTIONS_DATASET_ID, "hdb_resale_transactions.csv"),
        ("cpi", CPI_DATASET_ID, "cpi.csv"),
    ]:
        try:
            csv_bytes = _download_dataset_csv(dataset_id)
            _upload_to_s3(csv_bytes, filename)
            results[label] = {"status": "success", "bytes": len(csv_bytes)}
        except Exception as exc:
            logger.exception("Failed to refresh %s", label)
            results[label] = {"status": "error", "message": str(exc)}

    return {"statusCode": 200, "body": json.dumps(results)}