import os

import pandas as pd
import requests
import streamlit as st

# --- Config ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_BASE_URL}/predict"
EVALUATE_URL = f"{API_BASE_URL}/evaluate/"

st.set_page_config(page_title="Complaint Routing", layout="centered")
st.title("CFPB Complaint Routing")
st.caption("Automated routing of consumer complaints to the right internal team.")

tab1, tab2 = st.tabs(["Predict", "Model Evaluation"])

# ============================================================
# TAB 1: Live prediction
# ============================================================
with tab1:
    text = st.text_area(
        "Complaint narrative",
        height=150,
        placeholder="I do not recognize a charge on my credit card.",
    )
    model_choice = st.selectbox("Model", ["tfidf", "embeddings", "bert"])
    submit = st.button("Predict")

    if submit:
        if not text.strip():
            st.warning("Enter some complaint text first.")
        else:
            with st.spinner("Calling API..."):
                try:
                    response = requests.post(
                        PREDICT_URL,
                        params={"model": model_choice},
                        json={"text": text},
                        timeout=30,
                    )
                    response.raise_for_status()
                    data = response.json()
                except Exception:
                    st.error(
                        f"Request failed: {model_choice} model is not enabled on the free deployment."
                    )
                    st.stop()

            result = (
                data["results"][0]
                if isinstance(data["results"], list)
                else data["results"]
            )

            st.subheader(f"Predicted category: {result['prediction']}")

            col1, col2 = st.columns(2)
            col1.metric("Latency", result["latency"])
            col2.metric("Decision", result["decision"])

            probs_df = pd.DataFrame(
                sorted(
                    result["probabilities"].items(), key=lambda x: x[1], reverse=True
                ),
                columns=["Category", "Probability"],
            )
            st.bar_chart(probs_df.set_index("Category"))
            st.dataframe(probs_df, width="stretch", hide_index=True)

# ============================================================
# TAB 2: Run evaluation via API
# ============================================================
with tab2:
    st.header("Run Model Evaluation")

    dataset_location = st.text_input(
        "Dataset location",
        value="https://raw.githubusercontent.com/cappeadu/doc-routing-cfpb/refs/heads/dev/dataset/val.csv",
        help="Path to the CSV the API should evaluate against.",
    )
    checkpoint = st.text_input(
        "Checkpoint path",
        value="models/artifacts.joblib",
        help="Leave as default to evaluate the TF-IDF model.",
    )

    run_eval = st.button("Run Evaluation")

    if run_eval:
        with st.spinner("Running evaluation... this may take a minute"):
            try:
                response = requests.post(
                    EVALUATE_URL,
                    params={
                        "dataset_location": dataset_location,
                        "checkpoint": checkpoint,
                    },
                    timeout=600,
                )
                response.raise_for_status()
                eval_data = response.json()["results"]
            except Exception as e:
                st.error(f"Evaluation request failed: {e}")
                st.stop()

        overall = eval_data["overall"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{overall['accuracy']:.3f}")
        col2.metric("Macro F1", f"{overall['f1_macro']:.3f}")
        col3.metric("Weighted F1", f"{overall['f1_weighted']:.3f}")
        col4.metric("Samples", int(overall["num_samples"]))

        st.caption(
            f"Evaluated: {eval_data['time stamp']} · Took {eval_data['total time']}"
        )

        st.subheader("Per-Class Performance")
        per_class_df = pd.DataFrame(eval_data["per_class"]).T.reset_index()
        per_class_df.columns = ["Category", "Precision", "Recall", "F1", "Samples"]
        per_class_df = per_class_df.sort_values("F1", ascending=False)

        st.dataframe(
            per_class_df.style.format(
                {"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}
            ),
            width="stretch",
            hide_index=True,
        )

        st.bar_chart(per_class_df.set_index("Category")[["Precision", "Recall", "F1"]])
