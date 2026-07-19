"""
Local smoke test for the Lambda's download logic.

Run this on your own machine (needs real internet access to data.gov.sg,
which the sandbox this was written in does not have). Does not touch S3 or
require any AWS credentials -- it only exercises _download_dataset_csv.

Usage:
    python test_local_download.py
"""

from lambda_function import (
    CPI_DATASET_ID,
    TRANSACTIONS_DATASET_ID,
    _download_dataset_csv,
)


def smoke_test(label: str, dataset_id: str) -> None:
    print(f"\n--- {label} ({dataset_id}) ---")
    csv_bytes = _download_dataset_csv(dataset_id)
    print(f"Downloaded {len(csv_bytes):,} bytes")

    # Decode just enough to sanity-check it looks like a real CSV, without
    # printing the whole thing.
    text = csv_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    print(f"Row count (including header): {len(lines):,}")
    print(f"Header: {lines[0]}")
    print(f"First data row: {lines[1] if len(lines) > 1 else '(no data rows)'}")


if __name__ == "__main__":
    smoke_test("Transactions", TRANSACTIONS_DATASET_ID)
    smoke_test("CPI", CPI_DATASET_ID)