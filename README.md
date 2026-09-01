# Fake News Detection

This is an end-to-end Fake News Detection pipeline built using Natural Language Processing (NLP) techniques and Machine Learning. 

It satisfies the following guidelines:
* Uses a public dataset from Hugging Face (`GonzaloA/fake_news`).
* Applies NLP techniques like TF-IDF vectorization (with support for BERT).
* Uses classification models (SVM by default, structure for BERT included).
* Exposes a FastAPI application for real-time inference.

## Project Structure

```
FakeNewsDetection/
│
├── data/                  # Dataset (csv files, downloaded via script)
├── models/                # Saved models and vectorizers
├── notebooks/
│   └── EDA.ipynb          # Exploratory Data Analysis notebook
├── src/
│   ├── data_loader.py     # Script to download/load the dataset
│   ├── preprocess.py      # Text cleaning and TF-IDF logic
│   ├── model.py           # Model definitions (SVM, BERT)
│   └── train.py           # Main training script
├── app.py                 # FastAPI application for inference
├── requirements.txt       # Project dependencies
└── .gitignore
```

## Setup & Installation

1. **Clone the repository** (if on Github) and navigate to the folder:
   ```bash
   cd FakeNewsDetection
   ```

2. **Create a virtual environment and install dependencies**:

   **For Windows:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

   **For macOS and Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Install the requirements** (run this after activating the environment):
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

By default, the script trains a Support Vector Machine (SVM) using TF-IDF features.

1. Navigate to the `src` folder:
   ```bash
   cd src
   ```

2. Run the training script:
   ```bash
   python train.py --model svm
   ```
   This will download the dataset to `../data/`, preprocess the text, train an SVM, and save the model artifacts (vectorizer and classifier) to `../models/`.

*(Optional)* To train using BERT (requires `transformers` and `torch`):
```bash
python train.py --model bert
```

## Running the API

Once the model is trained and saved in the `models/` directory, you can start the API:

```bash
uvicorn app:app --reload
```

Then visit `http://127.0.0.1:8000/docs` in your browser to test the `/predict` endpoint interactively.

### Example API Request

```json
{
  "text": "Scientists have discovered a new species of glowing mushrooms in the Amazon rainforest."
}
```

## EDA

To view the Exploratory Data Analysis, open `notebooks/EDA.ipynb` in Jupyter or VSCode.

## Note
The dataset uses 0 for Real News and 1 for Fake News. The preprocessing removes special characters, URLs, and standardizes the text before feeding it to the TF-IDF vectorizer.

