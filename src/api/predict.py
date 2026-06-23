
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import numpy as np
import os

# Load Model and Tokenizer
model_id = "oguzhandurmaz/deberta-v3-small-cefr-lora-merged"
onnx_path = "onnx/model_quantized.onnx"

model_path = hf_hub_download(
    repo_id="oguzhandurmaz/deberta-v3-small-cefr-lora-merged",
    filename="onnx/model_quantized.onnx"
)

class CefrPredictorModel:

    def __init__(self,model_dir):
        self.model_dir = model_dir
        self._tokenizer = None
        self._session = None
    
    def load_model(self):
        onnx_path = os.path.join(self.model_dir, "onnx", "model_quantized.onnx")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    def predict(self,text):
    # Tokenize input
        inputs = self._tokenizer(text, return_tensors="np")
        # Convert inputs to int64 (required by ONNX)
        onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        
        # Run Inference
        logits = self._session.run(None, onnx_inputs)[0]
        
        # Softmax to get probabilities
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=-1)
        print("Probs: ",probs)
        print(probs[0][1])
        
        # CEFR Label Map
        labels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        print([ { "label": label, "confidence": round(float(probs[0][i]),1) } for i, label in enumerate(labels)])
        return [ { "label": label, "confidence": round(float(probs[0][i]),1) } for i, label in enumerate(labels)]




# Example Usage
#result = predict("The economic ramifications of the policy were far-reaching and complex.")
#print(f"Predicted Level: {result['label']} (Confidence: {result['confidence']:.2f})")