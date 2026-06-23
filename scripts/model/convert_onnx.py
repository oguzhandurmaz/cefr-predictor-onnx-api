import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

model_path = "models/cefr_deberta_lora"
save_dir = Path("models/onnx/cefr_deberta_lora_onnx")
save_dir.mkdir(exist_ok=True)

# Load Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create a dummy input for ONNX graph structure
dummy_model_input = tokenizer("This is a sample input", return_tensors="pt")

# Export to ONNX
torch.onnx.export(
    model, 
    tuple(dummy_model_input.values()),
    f=save_dir / "model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'}, 
        'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
        'logits': {0: 'batch_size'}
    },
    opset_version=14
)

# Copy tokenizer files to the same directory (will be needed for deployment)
tokenizer.save_pretrained(save_dir)

print(f"Successfully! Files are here: {save_dir}")