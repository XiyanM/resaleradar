"""
Monthly data refresh Lambda for ResaleRadar.

Triggered by an EventBridge scheduled rule. Downloads the latest HDB resale
transactions and CPI datasets from data.gov.sg and overwrites the raw-data/
files in S3. Both datasets are always downloaded as a full snapshot (not an
incremental diff), so a straight overwrite is correct -- no merge logic needed.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request

import boto3

import features
import retrain

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Fixed data.gov.sg dataset identifiers -- these do not change between runs,
# only the data behind them does. See conversation history for provenance.
CPI_DATASET_ID = "d_bdaff844e3ef89d39fceb962ff8f0791"
TRANSACTIONS_DATASET_ID = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"

API_HOST = "https://api-open.data.gov.sg"

REQUEST_HEADERS = {
    "Accept": "application/json",
    # Python's default User-Agent (Python-urllib/3.x) is blocked by
    # Cloudflare's bot protection in front of this API -- confirmed by
    # comparing a curl request (succeeds, distinct User-Agent) against the
    # original urllib request (403). Any non-default identifier works.
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
    """
    Run the initiate-download / poll-download sequence for one dataset and
    return the raw CSV bytes.
    """
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

    # The returned URL is a direct S3-hosted CSV link, not a data.gov.sg
    # endpoint -- fetching it does not count against the data.gov.sg rate limit.
    csv_req = urllib.request.Request(download_url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(csv_req, timeout=60) as csv_response:
        return csv_response.read()


def _upload_to_s3(csv_bytes: bytes, filename: str, bucket: str, prefix: str) -> None:
    key = f"{prefix}{filename}"
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=csv_bytes)
    logger.info("Uploaded %d bytes to s3://%s/%s", len(csv_bytes), bucket, key)


def _trigger_render_redeploy(hook_url: str) -> None:
    req = urllib.request.Request(hook_url, method="POST")
    with urllib.request.urlopen(req, timeout=15) as response:
        logger.info("Render redeploy triggered, status=%s", response.status)


def handler(event, context):
    # Read env vars here, not at module scope, so this file can be imported
    # for local testing (e.g. testing _download_dataset_csv directly)
    # without needing AWS credentials or these variables set.
    s3_bucket = os.environ["S3_BUCKET"]
    s3_prefix = os.environ.get("S3_PREFIX", "raw-data/")

    results = {}

    # Step 1: refresh raw data from data.gov.sg
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
            # If either raw dataset failed to refresh, there's no point
            # continuing to feature engineering or retraining -- both would
            # either use stale data or crash on a missing file.
            return {"statusCode": 500, "body": json.dumps(results)}

    # Step 2: rebuild features_with_geo.csv from the freshly refreshed raw data
    try:
        feature_result = features.rebuild_features_with_geo(s3_bucket)
        results["features"] = {"status": "success", **feature_result}
    except Exception as exc:
        logger.exception("Feature engineering failed")
        results["features"] = {"status": "error", "message": str(exc)}
        return {"statusCode": 500, "body": json.dumps(results)}

    # Step 3: retrain, only promoting if the new model beats the current champion
    try:
        retrain_result = retrain.retrain_and_maybe_promote(s3_bucket)
        results["retrain"] = {"status": "success", **retrain_result}

        if retrain_result.get("promoted"):
            hook_url = os.environ.get("RENDER_DEPLOY_HOOK_URL")
            if hook_url:
                try:
                    _trigger_render_redeploy(hook_url)
                    results["redeploy"] = {"status": "triggered"}
                except Exception as exc:
                    logger.exception("Failed to trigger Render redeploy")
                    results["redeploy"] = {"status": "error", "message": str(exc)}
            else:
                logger.warning("Model promoted but RENDER_DEPLOY_HOOK_URL not set -- skipping redeploy trigger")
                results["redeploy"] = {"status": "skipped", "reason": "no hook URL configured"}
    except Exception as exc:
        logger.exception("Retraining failed")
        results["retrain"] = {"status": "error", "message": str(exc)}

    return {"statusCode": 200, "body": json.dumps(results)}