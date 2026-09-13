from flask import Flask, render_template, request
import torch
import json
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


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
# HUGGING FACE MODEL
# ============================================================

MODEL_NAME = "VaishnaviC17/ev-policy-distilbert"


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
# LOAD TRAINED DISTILBERT MODEL
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()


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

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

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
