from langdetect import detect ,DetectorFactory
DetectorFactory.seed=0
def detect_lang(text:str)->str:
    try:
        lang = detect(text)
        if lang in ['en','ar']:
            print('Detected English language :'.format(lang))
            return lang
        else:
            print(f" Unsupported language detected. Defaulting to English...")
            return "en"
    except Exception as e:
        print(f" Unsupported language detected. Please try again later... Defaulting to English")
        return 'en'
from transformers import pipeline
print("[ANALYZER] Loading English model...")
english_classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Load the Arabic sentiment analysis model
print("[ANALYZER] Loading Arabic model...")
arabic_classifier = pipeline(
    "sentiment-analysis",
    model='aubmindlab/bert-base-arabertv02-twitter')
print(arabic_classifier("أنا سعيد جدًا اليوم"))

def get_sentiment(text:str,language:str)->dict:
    if language =='ar':
        result=arabic_classifier(text)[0]
        label =result['label'].replace("pos","Positive").replace("neg","Negative").replace("neutral","Neutral")
    else:
        result=english_classifier(text)[0]
        label =result['label']
    sentiment=label
    confidence =result['score']*100
    print(f" Sentiment: {sentiment} \n Confidence: {confidence:.1f}%")
    return {
        "sentiment":sentiment,
        "confidence":confidence
    }
import streamlit as st
st.set_page_config(
    page_title="Bilingual Sentiment Analysis",
    page_icon="🌍",
    layout='centered'
)
st.title("Arabic - English Sentiment Analyzer")
st.write("Enter text in either English or Arabic to analyze it's sentiment")
user_input =st.text_area('Enter Your TExt here....',height=250)
if st.button("Analyze Sentiment"):
    if user_input:
        with st.spinner('Analyzing Sentiment...'):
            language = detect_lang(user_input)
            analysis_result= get_sentiment(user_input,language)
            st.subheader("Analysis Result")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Language Detected  :", language)
            with col2:
                st.metric("Sentiment:", analysis_result['sentiment'])
            st.write(f" Confidence: {analysis_result['confidence']:.1f}%")
            with st.expander("See Details"):
                st.write(f" User Input : {user_input}")
                st.write(f" Language detected : {language}")
                st.write(f" Predicted Sentiment : {analysis_result['sentiment']}")
                st.write(f" Confidence : {analysis_result['confidence']:.1f}%")
    else:
        st.warning("Please enter some text to analyze...")

