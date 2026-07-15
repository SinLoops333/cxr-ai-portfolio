"""Streamlit UI for the CXR Copilot.

Talks to the FastAPI backend at ``COPILOT_API`` (default http://localhost:8000).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API = os.environ.get("COPILOT_API", "http://localhost:8000")

st.set_page_config(page_title="CXR Copilot", layout="wide")
st.title("Chest X-ray Copilot")
st.caption("Research prototype. Not a medical device. Do not use for clinical decisions.")

with st.sidebar:
    st.markdown(f"**Backend:** `{API}`")
    q = st.text_area("Question (optional)", "Summarize the key findings.")
    threshold_hint = st.info("Claims flagged `[UNVERIFIED]` failed both the classifier and RAG support checks.")

img_file = st.file_uploader("Chest X-ray (PNG/JPEG)", type=["png", "jpg", "jpeg"])
if img_file and st.button("Analyze"):
    st.image(img_file, caption="Input", width=400)
    with st.spinner("Running pipeline..."):
        r = requests.post(f"{API}/predict", files={"image": img_file.getvalue()}, data={"question": q}, timeout=180)
    if r.status_code != 200:
        st.error(f"API error {r.status_code}: {r.text}")
    else:
        data = r.json()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Classifier probabilities")
            st.dataframe({"probability": data["classifier"]}, use_container_width=True)
            st.markdown("**Asserted findings**")
            st.write(", ".join(data["asserted"]) or "none above threshold")
        with c2:
            st.subheader("Drafted report")
            st.text(data["draft"])
            st.subheader("Verified report")
            st.text(data["verified"]["revised"])
            st.subheader("Per-claim audit")
            st.dataframe(data["verified"]["claims"], use_container_width=True)
            st.subheader("Faithfulness stats")
            st.json(data["verified"]["stats"])
