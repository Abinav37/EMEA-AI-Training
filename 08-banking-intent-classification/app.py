from __future__ import annotations

import streamlit as st

from predict_intent import load_model_and_tokenizer, predict_with_loaded_model

LOW_CONFIDENCE_THRESHOLD = 0.5


EXAMPLE_QUERIES = [
    "I ordered a new card but it still has not arrived after two weeks.",
    "I need to verify my identity before my card can be used again.",
    "My direct debit payment was not recognised and I need help disputing it.",
    "A transfer failed and I want to know when the money will reach my account.",
    "My card payment was declined and I need to understand why.",
]


def _load_model_bundle():
    try:
        return load_model_and_tokenizer()
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        return {"error": str(exc)}


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    return _load_model_bundle()


def format_intent_label(label_name: str) -> str:
    return label_name.replace("_", " ").title()


def main() -> None:
    st.set_page_config(
        page_title="Banking Intent Classifier",
        page_icon="🏦",
        layout="centered",
    )

    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #f6fafc 0%, #eef4f9 100%);
        }
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
        }
        .banking-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: #123a5b;
            margin-bottom: 0.15rem;
        }
        .banking-subtitle {
            font-size: 1rem;
            color: #48657b;
            margin-bottom: 1.2rem;
        }
        .result-card {
            background: linear-gradient(135deg, #ffffff 0%, #edf6ff 100%);
            border: 1px solid #dfeaf3;
            border-radius: 16px;
            padding: 1.25rem 1.3rem;
            box-shadow: 0 8px 20px rgba(18, 58, 91, 0.06);
        }
        .intent-chip {
            display: inline-block;
            background: #123a5b;
            color: white;
            padding: 0.45rem 0.8rem;
            font-weight: 600;
            border-radius: 999px;
            margin-bottom: 0.5rem;
        }
        .warning-box {
            background: #fff8ea;
            border: 1px solid #f0d384;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            color: #7b5200;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="banking-title">Banking Intent Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="banking-subtitle">A lightweight demo for classifying customer banking queries with a saved DistilBERT model.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Project snapshot")
        st.info(
            "The app uses the saved model artifacts already present in this project. "
            "No retraining is performed here."
        )

        st.markdown("**Verified dataset sizes**")
        st.write("- Train: 10,003 rows")
        st.write("- Dev: 3,080 rows")
        st.write("- Test: 3,080 rows")
        st.write("- Classes: 77")

        st.markdown("**Verified lightweight evaluation metrics**")
        st.write("- Accuracy: 0.0162")
        st.write("- Macro precision: 0.0008")
        st.write("- Macro recall: 0.0162")
        st.write("- Macro F1: 0.0014")
        st.caption("Low-confidence threshold: below 50% (confidence < 0.5) is treated as uncertain.")
        st.caption(
            "These numbers come from the existing project log and should be treated as a demo baseline, not a production benchmark."
        )

    bundle = get_model_bundle()
    if bundle is None or "error" in bundle:
        st.error(
            "The saved model or tokenizer could not be loaded. Please check that the artifacts are present under "
            "08-banking-intent-classification/artifacts and that the project dependencies are installed."
        )
        if bundle and "error" in bundle:
            st.warning(bundle["error"])
        st.stop()

    model, tokenizer, device, label_map = bundle

    st.markdown("### Enter a banking query")
    st.caption("Low-confidence threshold: below 50% (confidence < 0.5) is treated as uncertain.")
    example_cols = st.columns(len(EXAMPLE_QUERIES))
    for index, example in enumerate(EXAMPLE_QUERIES):
        with example_cols[index]:
            if st.button(example[:38] + ("..." if len(example) > 38 else ""), key=f"example_{index}"):
                st.session_state["query_input"] = example

    query = st.text_area(
        "Customer message",
        key="query_input",
        height=160,
        placeholder="Type a customer query here, for example: 'I need to verify my identity before my card can be used again.'",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        predict_clicked = st.button("Predict Intent", type="primary")

    if predict_clicked:
        if not query or not query.strip():
            st.warning("Please enter a banking customer query before predicting.")
        else:
            try:
                with st.spinner("Tokenizing the query and running the model..."):
                    result = predict_with_loaded_model(
                        query,
                        model=model,
                        tokenizer=tokenizer,
                        label_map=label_map,
                        device=device,
                    )
                    label_name, confidence = result
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                st.error(f"Inference failed: {exc}")
                st.stop()
            except Exception as exc:  # pragma: no cover
                st.error(f"An unexpected inference error occurred: {exc}")
                st.stop()

            percentage = confidence * 100
            displayed_label = format_intent_label(label_name)
            st.markdown("### Prediction result")
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="intent-chip">{displayed_label}</div>
                    <div style="font-size: 2.1rem; font-weight: 700; color: #123a5b;">{percentage:.1f}% confidence</div>
                    <div style="font-size: 0.95rem; color: #48657b; margin-top: 0.3rem;">Model estimate: {confidence:.4f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(min(1.0, max(0.0, confidence)))
            st.caption("Confidence is the model's estimate, not a guarantee of correctness.")

            if confidence < LOW_CONFIDENCE_THRESHOLD:
                st.markdown(
                    f"""
                    <div class="warning-box">
                        Low confidence — this prediction may be incorrect. Threshold: {LOW_CONFIDENCE_THRESHOLD * 100:.0f}%.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.success("This prediction is above the low-confidence threshold, but it is still only a model estimate.")

    st.markdown("---")
    st.markdown("### How it works")
    st.write(
        "1. The app accepts a customer query in the text box.\n"
        "2. The text is tokenized so the transformer can read the input.\n"
        "3. DistilBERT performs inference on the encoded text.\n"
        "4. The model returns a predicted banking intent and a confidence estimate."
    )

    st.markdown("---")
    st.markdown("### Project information")
    st.write("- Dataset: real train/dev/test CSV files in the project folder")
    st.write("- Model: distilbert-base-uncased fine-tuned for banking intent classification")
    st.write("- Artifacts: saved under the project artifacts folder")
    st.write("- Inference style: CPU-first and Mac-friendly, using the saved model only")


if __name__ == "__main__":
    main()
