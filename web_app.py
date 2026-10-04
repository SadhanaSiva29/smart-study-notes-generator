"""
Web Application for Smart Study Notes Generator
===============================================
A modern, responsive Flask web application that allows users to paste text,
choose presets, adjust summary length, and visualize key study points,
reduction metrics, and vocabulary badges in real-time.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from smart_study_notes import SmartStudyNotesGenerator

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates")
)

# Initialize generator once during app startup
print("[*] Initializing generator for Web App...")
generator = SmartStudyNotesGenerator()

SAMPLE_PARAGRAPHS = {
    "healthcare_ai": {
        "title": "Artificial Intelligence in Healthcare",
        "category": "Computer Science & Medicine",
        "text": (
            "Artificial intelligence in healthcare is revolutionizing how medical professionals diagnose and "
            "treat complex diseases. Machine learning algorithms analyze intricate medical imaging datasets, "
            "such as MRI scans and radiological X-rays, with remarkable precision, often detecting subtle "
            "malignant lesions that could easily escape human observation. Furthermore, AI-powered predictive "
            "models assist hospitals in forecasting patient admission surges and optimizing vital resource "
            "allocation across intensive care units. Continuous remote patient monitoring through wearable biosensors "
            "enables timely clinical interventions for chronic condition management. Although these transformative "
            "technological advancements promise to elevate patient survival rates and lower administrative expenses, "
            "seamlessly integrating AI into clinical workflows necessitates addressing critical hurdles regarding "
            "patient data confidentiality, algorithmic bias, regulatory compliance, and physician oversight."
        )
    },
    "renewable_energy": {
        "title": "Renewable Energy Transition",
        "category": "Environmental Engineering",
        "text": (
            "The global transition toward renewable energy is accelerating rapidly in response to escalating climate "
            "change and the imperative to drastically reduce greenhouse gas emissions. Solar photovoltaic panels "
            "and wind turbines are now generating electricity at costs highly competitive with conventional fossil "
            "fuels, driving unprecedented investments across industrialized and developing economies alike. However, "
            "the inherent intermittency of wind and sunlight poses formidable engineering challenges for national "
            "electrical grids, requiring massive deployment of advanced grid-scale battery storage, pumped hydroelectric "
            "systems, and modernized high-voltage direct-current transmission networks. Transitioning away from "
            "legacy coal and gas infrastructure also carries significant socioeconomic repercussions, compelling "
            "policymakers to devise just transition frameworks that provide green workforce reskilling, energy "
            "equity, and economic diversification for displaced fossil fuel communities."
        )
    },
    "jwst_astronomy": {
        "title": "James Webb Space Telescope",
        "category": "Astrophysics & Space Exploration",
        "text": (
            "The James Webb Space Telescope represents an unprecedented technological leap in observational "
            "astronomy, offering humanity its deepest and most detailed view into the primordial cosmos. Equipped "
            "with high-resolution infrared detectors and an enormous gold-coated beryllium mirror, the observatory "
            "can pierce through dense interstellar dust clouds to observe the very first galaxies that coalesced just "
            "a few hundred million years following the Big Bang. In addition to probing cosmic origins, the telescope "
            "performs sophisticated atmospheric spectroscopy on exoplanets orbiting distant stellar systems, searching "
            "for molecular biosignatures such as water vapor, methane, and carbon dioxide. Operating at the Second "
            "Lagrange Point nearly one million miles from Earth, this flagship international mission is systematically "
            "dismantling long-held astrophysical assumptions while revolutionizing our understanding of stellar "
            "genesis and planetary habitability."
        )
    }
}


@app.route("/")
def index():
    return render_template("index.html", samples=SAMPLE_PARAGRAPHS)


@app.route("/api/samples", methods=["GET"])
def get_samples():
    return jsonify(SAMPLE_PARAGRAPHS)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "service": "smart-study-notes-generator", "model": generator.model_name})


@app.route("/api/summarize", methods=["POST"])
@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json(force=True) if (request.is_json or request.data) else {}
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"success": False, "error": "Please provide a paragraph to summarize."}), 400

        ratio = float(data.get("ratio", 0.40))
        # Ensure ratio stays in sensible range
        ratio = max(0.2, min(0.6, ratio))

        result = generator.generate_notes(text, target_summary_ratio=ratio)
        result["success"] = True
        result["reduction_percentage"] = result["percentage_reduction"]
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[+] Starting Smart Study Notes Generator Web App at http://127.0.0.1:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
