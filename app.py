import streamlit as st
from langdetect import detect, DetectorFactory
from transformers import pipeline

DetectorFactory.seed = 0


@st.cache_resource
def load_models():
    print("[LOADING] Loading English model...")
    english_classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

    print("[LOADING] Loading Arabic model...")
    arabic_classifier = pipeline(
        "sentiment-analysis",
        model="aubmindlab/bert-base-arabertv02-twitter",
    )
    return english_classifier, arabic_classifier


def detect_lang(text: str) -> str:
    try:
        language = detect(text)
        if language in ["en", "ar"]:
            return language
        return "en"
    except Exception:
        return "en"


def normalize_label(label: str, language: str) -> str:
    text = str(label).lower()

    if language == "ar":
        if "pos" in text or "label_1" in text or "positive" in text:
            return "Positive"
        if "neg" in text or "label_0" in text or "negative" in text:
            return "Negative"
        if "neutral" in text or "label_2" in text:
            return "Neutral"
        return label

    if "pos" in text or "positive" in text:
        return "Positive"
    if "neg" in text or "negative" in text:
        return "Negative"
    if "neutral" in text:
        return "Neutral"
    return label


def analyze_sentiment(text: str, language: str, english_model, arabic_model):
    if language == "ar":
        result = arabic_model(text, truncation=True)[0]
    else:
        result = english_model(text, truncation=True)[0]

    sentiment = normalize_label(result["label"], language)
    confidence = round(result["score"] * 100, 1)

    return {
        "language": language,
        "sentiment": sentiment,
        "confidence": confidence,
        "raw_label": result["label"],
    }


st.set_page_config(
    page_title="Bilingual Sentiment Analyzer",
    page_icon="🌍",
    layout="centered",
)

st.title("🌍 Bilingual Sentiment Analyzer")
st.caption("Detects English or Arabic text and predicts sentiment in real time.")

english_model, arabic_model = load_models()

with st.sidebar:
    st.subheader("Try a sample")
    st.write("- I love this app and it works perfectly!")
    st.write("- هذا التطبيق ممتاز جدًا!")
    st.write("- This is terrible and frustrating.")

user_input = st.text_area(
    "Enter text here",
    value="I love this app and it works perfectly!",
    height=220,
    placeholder="Type in English or Arabic...",
)

col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    english_sample = st.button("English Sample")
with col2:
    arabic_sample = st.button("Arabic Sample")
with col3:
    clear_button = st.button("Clear")

if english_sample:
    st.session_state.user_input = "I love this app and it works perfectly!"
if arabic_sample:
    st.session_state.user_input = "هذا التطبيق ممتاز جدًا!"
if clear_button:
    st.session_state.user_input = ""

if "user_input" in st.session_state:
    user_input = st.session_state.user_input

if st.button("Analyze Sentiment", type="primary"):
    if user_input and user_input.strip():
        with st.spinner("Analyzing sentiment..."):
            language = detect_lang(user_input)
            result = analyze_sentiment(user_input, language, english_model, arabic_model)

        st.subheader("Analysis Result")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Language", result["language"])
        with col_b:
            st.metric("Sentiment", result["sentiment"])

        st.metric("Confidence", f"{result['confidence']}%")

        with st.expander("View Details"):
            st.write(f"Input: {user_input}")
            st.write(f"Detected language: {result['language']}")
            st.write(f"Sentiment: {result['sentiment']}")
            st.write(f"Confidence: {result['confidence']}%")
            st.write(f"Raw model label: {result['raw_label']}")
    else:
        st.warning("Please enter some text to analyze.")
