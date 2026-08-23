# 🌍 Bilingual Sentiment Analyzer

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)

A Streamlit app that detects sentiment in **English** and **Arabic** text using HuggingFace models.

## ✨ Features
- Detects language automatically (`langdetect`)
- Runs sentiment analysis with:
  - English: `distilbert-base-uncased-finetuned-sst-2-english`
  - Arabic: `aubmindlab/bert-base-arabertv02-twitter`
- Interactive UI with Streamlit

## 🚀 Getting Started

### Installation
```bash
git clone https://github.com/pradeesh-kumar-n/bilingual-sentiment-analyzer.git
cd bilingual-sentiment-analyzer
pip install -r requirements.txt
