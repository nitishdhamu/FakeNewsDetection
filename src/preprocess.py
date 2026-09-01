import re
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def clean_text(text):
    """
    Cleans text by removing special characters, URLs, and making it lowercase.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_and_vectorize(train_texts, val_texts, test_texts, max_features=5000, save_dir="models"):
    """
    Applies TF-IDF vectorization to the datasets.
    """
    print("Cleaning text...")
    train_texts = [clean_text(t) for t in train_texts]
    val_texts = [clean_text(t) for t in val_texts]
    test_texts = [clean_text(t) for t in test_texts]
    
    print("Applying TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(save_dir, "tfidf_vectorizer.pkl"))
    print(f"Vectorizer saved to {save_dir}/tfidf_vectorizer.pkl")
    
    return X_train, X_val, X_test, vectorizer

