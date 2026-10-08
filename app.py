import streamlit as st
from langdetect import detect, DetectorFactory
from transformers import pipeline

DetectorFactory.seed = 0


@st.cache_resource
def load_models():
    """Load and cache models so they are not reloaded on every Streamlit rerun."""
    print("[ANALYZER] Loading English model...")
    english_classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

    print("[ANALYZER] Loading Arabic model...")
    arabic_classifier = pipeline(
        "sentiment-analysis",
        model="aubmindlab/bert-base-arabertv02-twitter",
    )
    return english_classifier, arabic_classifier


def detect_lang(text: str) -> str:
    try:
        lang = detect(text)
        if lang in ["en", "ar"]:
            return lang
        return "en"
    except Exception:
        return "en"


def normalize_label(raw_label: str, language: str) -> str:
    label = str(raw_label).upper().replace(" ", "_")

    if language == "ar":
        if label in ["POS", "POSITIVE", "LABEL_1", "LABEL_2"] or label.endswith("1"):
            return "Positive"
        if label in ["NEG", "NEGATIVE", "LABEL_0"] or label.endswith("0"):
            return "Negative"
        if "NEUTRAL" in label or label == "LABEL_2":
            return "Neutral"
        if "POS" in label:
            return "Positive"
        if "NEG" in label:
            return "Negative"
        if "NEUT" in label:
            return "Neutral"

    if label in ["POS", "POSITIVE", "LABEL_1"] or label.endswith("1"):
        return "Positive"
    if label in ["NEG", "NEGATIVE", "LABEL_0"] or label.endswith("0"):
        return "Negative"
    if "NEUTRAL" in label or label == "LABEL_2":
        return "Neutral"
    if "POS" in label:
        return "Positive"
    if "NEG" in label:
        return "Negative"

    return raw_label.title()


def get_sentiment(text: str, language: str, english_classifier, arabic_classifier) -> dict:
    if language == "ar":
        result = arabic_classifier(text, truncation=True)[0]
        label = normalize_label(result["label"], "ar")
    else:
        result = english_classifier(text, truncation=True)[0]
        label = normalize_label(result["label"], "en")

    confidence = result["score"] * 100
    return {"sentiment": label, "confidence": confidence}


english_classifier, arabic_classifier = load_models()

st.set_page_config(
    page_title="Bilingual Sentiment Analysis",
    page_icon="🌍",
    layout="centered",
)

st.title("Arabic - English Sentiment Analyzer")
st.write("Enter text in either English or Arabic to analyze its sentiment")

user_input = st.text_area("Enter your text here....", height=250)

if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner("Analyzing Sentiment..."):
            language = detect_lang(user_input)
            analysis_result = get_sentiment(user_input, language, english_classifier, arabic_classifier)

            st.subheader("Analysis Result")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Language Detected:", language)
            with col2:
                st.metric("Sentiment:", analysis_result["sentiment"])

            st.write(f"Confidence: {analysis_result['confidence']:.1f}%")

            with st.expander("See Details"):
                st.write(f"User Input: {user_input}")
                st.write(f"Language detected: {language}")
                st.write(f"Predicted Sentiment: {analysis_result['sentiment']}")
                st.write(f"Confidence: {analysis_result['confidence']:.1f}%")
    else:
        st.warning("Please enter some text to analyze...")
