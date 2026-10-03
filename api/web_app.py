from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.route("/api/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    try:
        summary_output = summarizer(text, max_length=120, min_length=30, do_sample=False)
        summary = summary_output[0]['summary_text']

        # Word counts
        original_word_count = len(text.split())
        summary_word_count = len(summary.split())
        reduction_percentage = ((original_word_count - summary_word_count) / original_word_count) * 100

        return jsonify({
            "summary": summary,
            "original_word_count": original_word_count,
            "summary_word_count": summary_word_count,
            "reduction_percentage": round(reduction_percentage, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
