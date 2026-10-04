"""
Smart Study Notes Generator - Streamlit Web Application
=======================================================
A modern, interactive UI for generating concise study notes, key points,
reduction metrics, and vocabulary badges using Falconsai/text_summarization.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from smart_study_notes import SmartStudyNotesGenerator

st.set_page_config(
    page_title="Smart Study Notes Generator",
    page_icon="📚",
    layout="wide"
)

SAMPLE_PARAGRAPHS = {
    "Healthcare AI": (
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
    ),
    "Renewable Energy": (
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
    ),
    "JWST Space Telescope": (
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

st.title("📚 Smart Study Notes Generator")
st.caption("AI-Powered Summarization & Study Takeaways using `Falconsai/text_summarization` (T5)")

@st.cache_resource
def load_generator():
    return SmartStudyNotesGenerator()

with st.spinner("Loading Generative AI model..."):
    generator = load_generator()

# Sidebar for presets & options
st.sidebar.header("⚙️ Settings & Presets")
preset_choice = st.sidebar.selectbox("Load Sample Paragraph", ["(Select preset)"] + list(SAMPLE_PARAGRAPHS.keys()))
ratio = st.sidebar.slider("Target Summary Ratio", min_value=0.2, max_value=0.6, value=0.4, step=0.05,
                          help="Lower values produce shorter, more condensed summaries.")

initial_text = ""
if preset_choice in SAMPLE_PARAGRAPHS:
    initial_text = SAMPLE_PARAGRAPHS[preset_choice]

input_text = st.text_area(
    "Paste your study text or academic paragraph below:",
    value=initial_text,
    height=200,
    placeholder="Type or paste a paragraph here..."
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    generate_btn = st.button("✨ Generate Notes", type="primary", use_container_width=True)

if generate_btn and input_text.strip():
    with st.spinner("Summarizing & extracting key points..."):
        try:
            result = generator.generate_notes(input_text, target_summary_ratio=ratio)

            # Quantitative Metrics
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Original Words", result["original_word_count"])
            m2.metric("Summary Words", result["summary_word_count"])
            m3.metric("Words Reduced", result["words_reduced"])
            m4.metric("Reduction", f"{result['percentage_reduction']}%")

            st.caption(f"⚡ Generated in {result['latency_seconds']}s using `{result['model_used']}` on `{generator.device.upper()}`")

            # Output Columns
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📝 Concise Summary")
                st.info(result["summary"])

            with c2:
                st.subheader("🎯 Key Study Takeaways")
                for pt in result["key_points"]:
                    st.markdown(f"- {pt}")

            if result.get("key_concepts"):
                st.subheader("🏷️ Key Concepts & Vocabulary")
                st.write(" ".join([f"`#{concept}`" for concept in result["key_concepts"]]))

        except Exception as e:
            st.error(f"Error generating notes: {e}")

elif generate_btn and not input_text.strip():
    st.warning("Please paste or type a paragraph to generate notes.")
