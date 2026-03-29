# ---------------------------------------------
# Upload model to Hugging Face Hub
# ---------------------------------------------

from huggingface_hub import HfApi, login
import os

# Login (only runs once, safe to keep)
#login()

# Initialize API
api = HfApi()

# Define correct repo ID (CASE SENSITIVE)
repo_id = "Karthickshiva07/engine-failure-model"

# Define model file path (robust)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "engine_failure_model.pkl")

# Upload model
api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="engine_failure_model.pkl",
    repo_id=repo_id,
    repo_type="model"
)

print("Model uploaded successfully!")