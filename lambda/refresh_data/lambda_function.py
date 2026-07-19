import json
import logging
import os
import time
import urllib.error
import urllib.request

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CPI_DATASET_ID = "d_bdaff844e3ef89d39fceb962ff8f0791"
TRANSACTIONS_DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"

API_HOST = "https://api-open.data.gov.sg"

REQUEST_HEADERS = {
    "Accept": "application/json",
    # non-default identifier to bypass Cloudflare
    "User-Agent": "ResaleRadar-Lambda/1.0 (+https://github.com/XiyanM/resaleradar)",
}


def _api_get(path: str, max_retries: int = 3, retry_wait_seconds: int = 15) -> dict:
    url = API_HOST + path
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(url, headers=REQUEST_HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_retries:
                logger.info(
                    "Rate limited on %s, waiting %ss before retry %s/%s",
                    path, retry_wait_seconds, attempt + 1, max_retries,
                )
                time.sleep(retry_wait_seconds)
                continue
            raise


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

    csv_req = urllib.request.Request(download_url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(csv_req, timeout=60) as csv_response:
        return csv_response.read()


def _upload_to_s3(csv_bytes: bytes, filename: str, bucket: str, prefix: str) -> None:
    key = f"{prefix}{filename}"
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes)
    logger.info("Uploaded %d bytes to s3://%s/%s", len(csv_bytes), bucket, key)


def handler(event, context):
    s3_bucket = os.environ["S3_BUCKET"]
    s3_prefix = os.environ.get("S3_PREFIX", "raw-data/")

    results = {}

    for label, dataset_id, filename in [
        ("transactions", TRANSACTIONS_DATASET_ID, "hdb_resale_transactions.csv"),
        ("cpi", CPI_DATASET_ID, "cpi.csv"),
    ]:
        try:
            csv_bytes = _download_dataset_csv(dataset_id)
            _upload_to_s3(csv_bytes, filename, s3_bucket, s3_prefix)
            results[label] = {"status": "success", "bytes": len(csv_bytes)}
        except Exception as exc:
            logger.exception("Failed to refresh %s", label)
            results[label] = {"status": "error", "message": str(exc)}

    return {"statusCode": 200, "body": json.dumps(results)}