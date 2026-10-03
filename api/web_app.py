from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.route("/api/summarize", methods=["POST"])
def summarize():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        summary = summarizer(text, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
