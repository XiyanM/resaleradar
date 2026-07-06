
from sagemaker.xgboost.estimator import XGBoost

# ── Config ────────────────────────────────────────────────────────────────────

BUCKET = "resaleradar"
ROLE_ARN = "arn:aws:iam::942226532349:role/resaleradar-sagemaker-execution-role"


estimator = XGBoost(
    entry_point="train.py",          
    source_dir="scripts",            
    role=ROLE_ARN,                   
    instance_type="ml.m5.large",     
    instance_count=1,
    framework_version="1.7-1",       
    output_path=f"s3://{BUCKET}/training-jobs/",   
    base_job_name="resaleradar-baseline-training",  
)


estimator.fit(
    inputs={"train": f"s3://{BUCKET}/processed-data/"},
    wait=True,   
)

print("Training job complete.")
print(f"Model artifacts saved to: {estimator.model_data}")