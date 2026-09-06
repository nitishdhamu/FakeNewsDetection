# 📰 Fake News Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Enabled-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Enabled-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

Welcome to the **Fake News Detection** pipeline! This repository houses an end-to-end Machine Learning solution designed to solve one of the biggest problems in modern media: **Identifying Misinformation**.

By analyzing the textual content of news articles, this tool uses Natural Language Processing (NLP) techniques to predict whether a given news article is Fake or Real.

---

## ⚙️ What Does This Project Do?

1. **Fetches Public Datasets**: It automatically connects to the Hugging Face hub to download the `GonzaloA/fake_news` dataset and extracts testing, training, and validation samples.
2. **Trains Predictive Models**: It cleans the text (removing URLs, special characters), applies NLP techniques like TF-IDF vectorization, and trains robust models:
   - **Support Vector Machine (SVM)**: A highly effective statistical baseline model (Default).
   - **BERT (Transformers)**: A state-of-the-art Deep Learning sequence classification model.
3. **Real-World Inference**: It exposes the trained models via a production-grade **FastAPI** web application, allowing you to instantly classify any custom news text in real-time.

---

## 📖 Quick Start Guide

### Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/nitishdhamu/FakeNewsDetection.git
cd FakeNewsDetection
```

### 2. Set Up a Virtual Environment (Recommended)
**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Machine Learning Pipeline (Training)
This script orchestrates the entire pipeline: downloading the data to `data/`, preprocessing the text, training the SVM model, and saving the model artifacts (vectorizer and classifier) to `models/`.

```bash
python src/train.py --model svm
```

*(Optional)* To train using BERT instead (requires `transformers` and `torch`):
```bash
python src/train.py --model bert
```

### 5. Predict Real-World Articles (Inference API)
Once the model is trained and saved in the `models/` directory, you can start the real-time API server!

```bash
uvicorn app:app --reload
```

Then visit `http://127.0.0.1:8000/docs` in your browser to interactively test the `/predict` endpoint.

**Example API Requests:**

*Test 1: Real News*
```json
{
  "text": "The Federal Reserve announced on Wednesday that it will raise interest rates by 0.25% in an effort to combat inflation."
}
```

*Test 2: Fake News*
```json
{
  "text": "Pope Francis shocks the world by officially endorsing Donald Trump for President in the upcoming election."
}
```

### 6. Exploratory Data Analysis (EDA)
To view the Exploratory Data Analysis, open `notebooks/EDA.ipynb` in Jupyter or VSCode.

---
*Created as part of a Data Science Internship Project.*
