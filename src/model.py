from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

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
