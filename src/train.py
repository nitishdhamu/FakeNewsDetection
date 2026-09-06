import os
import argparse
from data_loader import load_fake_news_dataset
from preprocess import preprocess_and_vectorize
from model import train_svm, evaluate_model

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
