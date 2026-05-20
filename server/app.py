import os
import re
import csv
import warnings
import torch
import pandas as pd
from difflib import SequenceMatcher
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BlenderbotSmallTokenizer,
    BlenderbotSmallForConditionalGeneration
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# -----------------------------
# Config
# -----------------------------
SENTIMENT_MODEL_PATH = "./fine_tuned_sentiment_model"
RESPONSES_FILE = "custom_responses.csv"
FEEDBACK_FILE = "feedback_log.csv"

id2label = {0: "negative", 1: "neutral", 2: "positive"}

# -----------------------------
# Load Sentiment Model
# -----------------------------
print("Loading fine-tuned sentiment model...")
tokenizer = BertTokenizer.from_pretrained(SENTIMENT_MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
model.eval()

# -----------------------------
# Load BlenderBot
# -----------------------------
print("Loading BlenderBot...")
blender_tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
blender_model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")

# -----------------------------
# Ensure CSV files exist
# -----------------------------
if not os.path.exists(RESPONSES_FILE):
    pd.DataFrame(columns=["emotion", "chat", "response"]).to_csv(RESPONSES_FILE, index=False)

if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_input", "bot_response", "sentiment", "feedback"])


# -----------------------------
# Helpers
# -----------------------------
def load_custom_responses():
    try:
        return pd.read_csv(RESPONSES_FILE).dropna()
    except Exception:
        return pd.DataFrame(columns=["emotion", "chat", "response"])


def detect_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_class_id = outputs.logits.argmax().item()
    return id2label.get(predicted_class_id, "neutral")


def generate_blender_response(user_input):
    inputs = blender_tokenizer([user_input], return_tensors="pt")
    reply_ids = blender_model.generate(
        **inputs,
        max_length=60,
        do_sample=True,
        top_k=30,
        top_p=0.9,
        temperature=0.6,
        repetition_penalty=1.2
    )
    return blender_tokenizer.decode(reply_ids[0], skip_special_tokens=True)


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def find_custom_response(user_input, sentiment):
    custom_responses = load_custom_responses()
    filtered = custom_responses[custom_responses["emotion"].str.lower() == sentiment.lower()]

    best_match = None
    best_score = 0

    for row in filtered.itertuples():
        score = similarity(user_input, row.chat)
        if score > best_score:
            best_score = score
            best_match = row

    if best_match and best_score > 0.6:
        return best_match.response

    return None


def is_simple_greeting(text):
    cleaned = text.strip().lower()
    return cleaned in ["hi", "hello", "hey", "hi!", "hello!", "hey!"]


# -----------------------------
# Routes
# -----------------------------
@app.route("/chat", methods=["POST"])
def chat_handler():
    try:
        data = request.get_json(silent=True) or {}
        user_input = (data.get("text") or data.get("message") or "").strip()

        if not user_input:
            return jsonify({"error": "Empty input"}), 400

        if is_simple_greeting(user_input):
            return jsonify({
                "response": "Hi there! How are you feeling today?",
                "sentiment": "neutral",
                "question": "How are you feeling today?"
            })

        sentiment = detect_sentiment(user_input)
        custom_response = find_custom_response(user_input, sentiment)

        final_response = custom_response if custom_response else generate_blender_response(user_input)

        return jsonify({
            "response": final_response,
            "sentiment": sentiment,
            "question": "Would you like to talk more about it?"
        })

    except Exception as e:
        return jsonify({
            "error": "Something went wrong while processing your request.",
            "details": str(e)
        }), 500


@app.route("/feedback", methods=["POST"])
def log_feedback():
    try:
        data = request.get_json(silent=True) or {}
        user_input = data.get("user_input", "")
        bot_response = data.get("bot_response", "")
        sentiment = data.get("sentiment", "")
        feedback = data.get("feedback", "")

        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([user_input, bot_response, sentiment, feedback])

        return jsonify({"status": "Feedback logged"}), 200

    except Exception as e:
        return jsonify({
            "error": "Failed to log feedback.",
            "details": str(e)
        }), 500


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Sentiment-Aware Chatbot Backend is running"}), 200


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)