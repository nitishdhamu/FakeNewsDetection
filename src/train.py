import os
import argparse
from data_loader import load_fake_news_dataset
from preprocess import preprocess_and_vectorize
from model import train_svm, evaluate_model, HAS_BERT

def main(model_type):
    print(f"--- Fake News Detection Training ({model_type.upper()}) ---")
    
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
    
    if model_type == 'svm':
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
        
    elif model_type == 'bert':
        if not HAS_BERT:
            print("Error: transformers and torch are required for BERT.")
            return
        
        from model import train_bert
        # Subsample for faster training in example
        model, tokenizer = train_bert(
            train_texts[:2000], train_labels[:2000], 
            val_texts[:500], val_labels[:500], 
            save_dir=os.path.join(models_dir, "bert_model")
        )
        print("BERT training complete. Evaluation omitted in this script for brevity.")
        
    else:
        print(f"Unknown model type: {model_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="svm", choices=['svm', 'bert'], help="Model type to train")
    args = parser.parse_args()
    main(args.model)
