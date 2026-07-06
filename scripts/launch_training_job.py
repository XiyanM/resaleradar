
from sagemaker.core import image_uris
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import InputData, Compute, SourceCode, OutputDataConfig

# ── Config ────────────────────────────────────────────────────────────────────

BUCKET = "resaleradar"
REGION = "ap-southeast-1"
ROLE_ARN = "arn:aws:iam::942226532349:role/resaleradar-sagemaker-execution-role"


xgb_image_uri = image_uris.retrieve(
    framework="xgboost",
    region=REGION,
    version="1.7-1",
    instance_type="ml.m5.large",
)


source_code = SourceCode(
    source_dir="scripts",
    entry_script="train.py",
)

# ── Compute config ────────────────────────────────────────────────────────────

compute = Compute(
    instance_type="ml.m5.large",
    instance_count=1,
)

# ── ModelTrainer ───────────────────────────────────────────────────────────────
model_trainer = ModelTrainer(
    training_image=xgb_image_uri,
    source_code=source_code,
    compute=compute,
    role=ROLE_ARN,
    output_data_config=OutputDataConfig(s3_output_path=f"s3://{BUCKET}/training-jobs/"),
    base_job_name="resaleradar-baseline-training",
)

# ── Input data ─────────────────────────────────────────────────────────────────

input_data = InputData(
    channel_name="train",
    data_source=f"s3://{BUCKET}/processed-data/",
)

# ── Launch ─────────────────────────────────────────────────────────────────────

model_trainer.train(
    input_data_config=[input_data],
    wait=True,   
)

print("Training job complete.")