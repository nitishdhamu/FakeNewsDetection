import os
import argparse
import joblib
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

from data_loader import load_fake_news_dataset
from preprocess import preprocess_and_vectorize

def train_svm(X_train, y_train, save_dir="models"):
    """
    Trains a Support Vector Machine (LinearSVC) model.
    """
    print("Training SVM model...")
    model = LinearSVC(random_state=42)
    model.fit(X_train, y_train)
    
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, "svm_model.pkl"))
    print(f"SVM Model saved to {save_dir}/svm_model.pkl")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model.
    """
    print("Evaluating model...")
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:")
    print(classification_report(y_test, predictions))

def main():
    print("--- Fake News Detection Training (SVM) ---")
    
    # Resolve absolute paths based on this script's location
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    models_dir = os.path.join(project_root, 'models')
    
    # 1. Load Data
    train_df, val_df, test_df = load_fake_news_dataset(save_dir=data_dir)
    
    # Using 'text' column for training, 'label' for targets (0: Real, 1: Fake)
    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    
    val_texts = val_df['text'].tolist()
    val_labels = val_df['label'].tolist()
    
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    # 2. Preprocess & Vectorize
    X_train, X_val, X_test, vectorizer = preprocess_and_vectorize(
        train_texts, val_texts, test_texts, save_dir=models_dir
    )
    
    # 3. Train
    model = train_svm(X_train, train_labels, save_dir=models_dir)
    
    # 4. Evaluate
    print("\n--- Validation Set Evaluation ---")
    evaluate_model(model, X_val, val_labels)
    
    print("\n--- Test Set Evaluation ---")
    evaluate_model(model, X_test, test_labels)

if __name__ == "__main__":
    main()
