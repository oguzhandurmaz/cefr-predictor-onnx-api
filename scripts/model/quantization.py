from onnxruntime.quantization import quantize_dynamic, QuantType

# Source Model
model_fp32 = "models/onnx/cefr_deberta_lora_onnx/model.onnx"
# Target Model
model_int8 = "models/onnx/cefr_deberta_lora_onnx/model_quantized.onnx"

# Start Dynamic Quantization
quantize_dynamic(
    model_input=model_fp32,
    model_output=model_int8,
    weight_type=QuantType.QInt8
)

print("Quantization completed!")