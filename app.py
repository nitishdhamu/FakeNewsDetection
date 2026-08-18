import os
import sys
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# add src to path to import preprocess functions
sys.path.append(os.path.join(BASE_DIR, 'src'))
from preprocess import clean_text

app = FastAPI(title="Fake News Detection API", description="API to classify news articles as fake or real.")

vectorizer = None
svm_model = None

@app.on_event("startup")
def load_models():
    global vectorizer, svm_model
    try:
        vec_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
        model_path = os.path.join(MODEL_DIR, "svm_model.pkl")
        vectorizer = joblib.load(vec_path)
        svm_model = joblib.load(model_path)
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load models. Did you train them? Error: {e}")

class NewsRequest(BaseModel):
    text: str

class NewsResponse(BaseModel):
    prediction: str
    confidence: float = 1.0

@app.post("/predict", response_model=NewsResponse)
def predict_news(request: NewsRequest):
    if vectorizer is None or svm_model is None:
        return {"prediction": "Model not loaded", "confidence": 0.0}
    
    # clean and vectorize input text
    cleaned = clean_text(request.text)
    vec_text = vectorizer.transform([cleaned])
    
    # model prediction: 0 is Real, 1 is Fake
    pred = int(svm_model.predict(vec_text)[0])
    label = "Fake" if pred == 1 else "Real"
    
    return {"prediction": label, "confidence": 1.0}

@app.get("/")
def read_root():
    return {"message": "Fake News Detection API is running. Go to /docs to test predictions."}
