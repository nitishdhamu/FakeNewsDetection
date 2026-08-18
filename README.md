# Fake News Detection

A machine learning project to detect whether news articles are real or fake using NLP and Support Vector Machines.

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## Running
Train the model:
```bash
python src/train_models.py
```

Run API:
```bash
uvicorn app:app --reload
```
