import os
import pandas as pd
from datasets import load_dataset

def load_fake_news_dataset(save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "train.csv")
    val_path = os.path.join(save_dir, "val.csv")
    test_path = os.path.join(save_dir, "test.csv")
    
    # use local files if already downloaded
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print(f"Loading cached dataset from {save_dir}/...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        return train_df, val_df, test_df
    
    # otherwise download from huggingface
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("GonzaloA/fake_news")
    
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])
    
    # save csv files for eda and offline use
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Dataset saved to {save_dir}/")
    return train_df, val_df, test_df

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, 'data')
    load_fake_news_dataset(save_dir=data_dir)
