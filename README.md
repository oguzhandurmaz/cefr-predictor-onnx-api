### 🚀 CEFR Predictor: AI-Powered English Proficiency Estimator

This project is a high-performance, production-ready API solution that estimates the **CEFR (Common European Framework of Reference for Languages)** level (A1-C2) of any given English text.

It utilizes a fine-tuned **DeBERTa-V3** model, optimized with **ONNX Quantization** techniques for maximum speed and efficiency.

---

## ✨ Key Features

- **🎯 High Accuracy:** Advanced model based on `DeBERTa-V3-Small` architecture, fine-tuned using LoRA.
- **⚡ Ultra Fast:** Millisecond-level inference on CPU via ONNX Runtime and INT8 Quantization.
- **🐳 Dockerized:** FastAPI-based architecture that can be deployed in seconds.
- **🔄 Auto Model Management:** Automatically downloads and caches the model from Hugging Face on the first run.

---

## 🛠️ Technical Stack

- **Model:** `oguzhandurmaz/deberta-v3-small-cefr-lora-merged`
- **Optimization:** ONNX (INT8 Quantization)
- **Framework:** FastAPI
- **Environment:** Python 3.13 / Docker

---

## 🚀 Quick Start

### 1. Local Installation

First, install the dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
cd src/api
fastapi dev app.py
```

### 2. Running with Docker

Run the application without worrying about local dependencies:

```bash
docker build -t cefr-predictor-api src/api/
docker run -p 80:80 cefr-predictor-api
```

---

## 🛰️ API Usage

### Prediction Request
Once the API is running, you can analyze text by sending a POST request to the `/predict` endpoint.

**Example Request (cURL):**
```bash
curl -X 'POST' \
  'http://localhost/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "The economic ramifications of the policy were far-reaching and complex."
}'
```

**Example Response:**
```json
[
  {"label": "A1", "confidence": 0.0},
  {"label": "A2", "confidence": 0.0},
  {"label": "B1", "confidence": 0.0},
  {"label": "B2", "confidence": 0.1},
  {"label": "C1", "confidence": 0.8},
  {"label": "C2", "confidence": 0.1}
]
```

---

## 📁 Project Structure

```text
.
├── src/api/
│   ├── app.py              # FastAPI Application
│   ├── predict.py          # ONNX Inference Logic
│   ├── download_model.py   # Hugging Face Model Downloader
│   ├── Dockerfile          # Container Configuration
│   └── requirements.txt    # API Dependencies
├── scripts/                # Training and Conversion Scripts
└── requirements.txt        # General Project Requirements
```

---

## 🧠 Model Details

The model analyzes complex sentence structures and vocabulary to categorize text into one of the following levels:
- **A1/A2:** Basic User
- **B1/B2:** Independent User
- **C1/C2:** Proficient User

---

## 🤝 Contributing

1. Fork this repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

**⭐ If you find this project useful, don't forget to give it a star!**
