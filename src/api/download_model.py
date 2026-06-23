import os
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer

def download_model_locally(repo_id, local_dir):
    # Predict ONNX file path
    onnx_local_path = os.path.join(local_dir, "onnx", "model_quantized.onnx")
    tokenizer_config = os.path.join(local_dir, "tokenizer_config.json")

    # Control if model is already downloaded
    if os.path.exists(onnx_local_path) and os.path.exists(tokenizer_config):
        print(f"✅ Model already exists in '{local_dir}'. Skipping download.")
        return local_dir

    print(f"🚀 Model not found locally or missing. Starting download: {repo_id}")
    
    # Create directory if not exists
    os.makedirs(local_dir, exist_ok=True)

    # 1. Download and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    tokenizer.save_pretrained(local_dir)

    # 2. Download ONNX model
    hf_hub_download(
        repo_id=repo_id,
        filename="onnx/model_quantized.onnx",
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )
    
    print(f"Download completed successfully: {local_dir}")
    return local_dir