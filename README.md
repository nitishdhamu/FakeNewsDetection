# Fake News Detection

A machine learning project that classifies news articles as either Real or Fake using Natural Language Processing (NLP) and a Support Vector Machine (LinearSVC) model.

The project uses the `GonzaloA/fake_news` dataset from Hugging Face, cleans and vectorizes the text using TF-IDF, and serves the trained model using a FastAPI backend.

## Project Structure

```text
FakeNewsDetection/
│
├── data/                  # Dataset folder (created automatically on download)
├── models/                # Saved model and vectorizer .pkl files
├── notebooks/
│   └── EDA.ipynb          # Exploratory data analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Downloads and splits dataset from Hugging Face
│   ├── preprocess.py      # Text cleaning and TF-IDF vectorization
│   └── train_models.py    # Trains LinearSVC model and saves artifacts
├── app.py                 # FastAPI application for predictions
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/nitishdhamu/FakeNewsDetection.git
cd FakeNewsDetection
```

### 2. Create and activate a virtual environment
For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

For macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## How to Run

### 1. Train the model
Run the training script to download the data, clean the text, train the LinearSVC model, and save the model artifacts:
```bash
python src/train_models.py
```
This will save `tfidf_vectorizer.pkl` and `svm_model.pkl` in the `models/` directory.

### 2. Start the FastAPI server
Once the models are saved, start the API:
```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` in your browser to test the interactive Swagger UI.

## Testing with Sample News

You can test the `/predict` endpoint directly in the Swagger UI (`/docs`) or by sending a POST request to `http://127.0.0.1:8000/predict`.

### Example 1: Real News
```json
{
  "text": "The Federal Reserve announced on Wednesday that it will raise interest rates by 0.25% in an effort to combat inflation."
}
```

### Example 2: Fake News
```json
{
  "text": "Pope Francis shocks the world by officially endorsing Donald Trump for President in the upcoming election."
}
```

## Exploratory Data Analysis
You can open `notebooks/EDA.ipynb` in Jupyter Notebook or VS Code to see class distribution and text length plots.
