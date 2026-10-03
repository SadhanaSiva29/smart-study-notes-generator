import streamlit as st
from transformers import pipeline

st.title("Smart Study Notes Generator")

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

text = st.text_area("Paste your paragraph here:")

if st.button("Summarize") and text:
    summarizer = load_summarizer()
    summary = summarizer(text, max_length=120, min_length=30, do_sample=False)[0]['summary_text']

    st.subheader("Summary")
    st.write(summary)

    # Word counts
    orig_count = len(text.split())
    sum_count = len(summary.split())
    reduction = round((orig_count - sum_count) / orig_count * 100, 2)

    st.write(f"Original words: {orig_count}")
    st.write(f"Summary words: {sum_count}")
    st.write(f"Reduction: {reduction}%")
