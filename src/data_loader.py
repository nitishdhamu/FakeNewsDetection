import pandas as pd
from datasets import load_dataset
import os

def load_fake_news_dataset(save_dir="data"):
    """
    Loads the fake news dataset from Hugging Face datasets.
    Dataset: GonzaloA/fake_news
    It contains 'title', 'text', 'label' (0 for True, 1 for Fake).
    """
    print("Loading dataset from Hugging Face...")
    # Load dataset
    dataset = load_dataset("GonzaloA/fake_news")
    
    # Convert to pandas dataframes
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save to CSV for EDA and other uses
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)
    
    print(f"Dataset downloaded and saved to {save_dir}/")
    return train_df, val_df, test_df

if __name__ == "__main__":
    load_fake_news_dataset("../data")

