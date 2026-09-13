from flask import Flask, render_template, request
import json
import os

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DASHBOARD_DATA_PATH = os.path.join(
    BASE_DIR,
    "results",
    "dashboard_data.json"
)


# ============================================================
# HUGGING FACE QUANTIZED MODEL
# ============================================================

MODEL_REPO = "VaishnaviC17/ev-policy-distilbert-quantized"


# ============================================================
# LOAD DASHBOARD DATA
# ============================================================

with open(
    DASHBOARD_DATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    dashboard_data = json.load(file)


# ============================================================
# DOWNLOAD / LOAD TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_REPO
)


# ============================================================
# DOWNLOAD QUANTIZED ONNX MODEL
# ============================================================

MODEL_PATH = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="model_quantized.onnx"
)


# ============================================================
# LOAD ONNX MODEL
# ============================================================

onnx_session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)


# ============================================================
# SENTIMENT LABELS
# ============================================================

LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


# ============================================================
# SENTIMENT PREDICTION
# ============================================================

def predict_sentiment(text):

    # Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Prepare inputs for ONNX Runtime
    onnx_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }

    # Run model
    outputs = onnx_session.run(
        None,
        onnx_inputs
    )

    # Get logits
    logits = outputs[0]

    # Convert logits to probabilities
    exp_logits = np.exp(
        logits - np.max(logits, axis=1, keepdims=True)
    )

    probabilities = (
        exp_logits /
        np.sum(exp_logits, axis=1, keepdims=True)
    )

    # Get predicted class
    predicted_class = int(
        np.argmax(
            probabilities,
            axis=1
        )[0]
    )

    # Get confidence
    confidence = float(
        probabilities[
            0,
            predicted_class
        ]
    )

    sentiment = LABELS[predicted_class]

    return sentiment, round(
        confidence * 100,
        2
    )


# ============================================================
# HOME — SENTIMENT ANALYZER
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = None
    confidence = None
    comment = ""

    if request.method == "POST":

        comment = request.form.get(
            "comment",
            ""
        ).strip()

        if comment:

            sentiment, confidence = predict_sentiment(
                comment
            )

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        comment=comment
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
def dashboard():

    return render_template(
        "dashboard.html",
        data=dashboard_data
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

@app.route("/comparison")
def comparison():

    return render_template(
        "comparison.html",
        data=dashboard_data
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
