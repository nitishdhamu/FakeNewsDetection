from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import sys

# Add src to path so we can import preprocess
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from preprocess import clean_text

app = FastAPI(title="Fake News Detection API", description="API to classify news articles as fake or real.")

# Load models at startup
MODEL_DIR = "models"
vectorizer = None
svm_model = None

import webbrowser
from threading import Timer

@app.on_event("startup")
def load_models():
    global vectorizer, svm_model
    try:
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        svm_model = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
        print("Models loaded successfully.")
        
        # Automatically open the browser to the /docs page after 1.5 seconds
        Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000/docs")).start()
    except Exception as e:
        print(f"Warning: Could not load models. Did you train them? Error: {e}")

class NewsRequest(BaseModel):
    text: str

class NewsResponse(BaseModel):
    prediction: str
    confidence: float = 1.0 # SVM doesn't natively output probabilities easily without Platt scaling, so dummy value

@app.post("/predict", response_model=NewsResponse)
def predict_news(request: NewsRequest):
    if not vectorizer or not svm_model:
        return {"prediction": "Model not loaded", "confidence": 0.0}
    
    # Preprocess
    cleaned = clean_text(request.text)
    
    # Vectorize
    vec_text = vectorizer.transform([cleaned])
    
    # Predict (0 is Real, 1 is Fake for this dataset)
    pred = svm_model.predict(vec_text)[0]
    
    label = "Fake" if pred == 1 else "Real"
    return {"prediction": label, "confidence": 1.0}

@app.get("/")
def read_root():
    return {"message": "Welcome to Fake News Detection API. Use /docs to test the API."}

