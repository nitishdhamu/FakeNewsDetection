from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# For BERT (optional, requires torch and transformers)
try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
    from torch.utils.data import Dataset
    HAS_BERT = True
except ImportError:
    HAS_BERT = False

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

if HAS_BERT:
    class NewsDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    def train_bert(train_texts, train_labels, val_texts, val_labels, save_dir="models/bert_model"):
        """
        Trains a BERT model for sequence classification.
        """
        print("Training BERT model... This may take a while.")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)
        
        train_dataset = NewsDataset(train_encodings, train_labels)
        val_dataset = NewsDataset(val_encodings, val_labels)
        
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch"
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        trainer.train()
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"BERT model saved to {save_dir}")
        return model, tokenizer

