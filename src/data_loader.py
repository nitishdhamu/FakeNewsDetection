import os
import pandas as pd
from datasets import load_dataset

def load_fake_news_dataset(save_dir="data"):
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("GonzaloA/fake_news")
    
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])
    
    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "test.csv"), index=False)
    
    print(f"Dataset saved to {save_dir}/")
    return train_df, val_df, test_df

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    load_fake_news_dataset(save_dir=data_dir)
