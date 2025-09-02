import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def download_and_unzip_spam_data(
    url,
    zip_path,
    extracted_path,
    data_file_path
):
    if data_file_path.exists():
        return
    response = urllib.request.urlopen(url)
    with open(zip_path,'wb') as out_file:
        out_file.write(response.read())
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    original_file_path = Path(extracted_path)/"SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state = 123
    )
    balanced_df = pd.concat([
        ham_subset, df[df['Label'] == "spam"]
    ])
    return balanced_df

def random_split(df, train_frac, validation_frac):
    df = df.sample(
        frac=1, random_state = 123
    ).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    val_end = train_end + int(len(df)*validation_frac)
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    return train_df, val_df, test_df

class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None,
                 pad_token_id = 50256):
        self.data = pd.read_csv(csv_file)
        
        self.encoded_texts = [
            tokenizer.encode(text)
            for text in self.data['Text']
        ]
        
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            self.encoded_texts = [
                encoded_text[:self.max_length] 
                for encoded_text in self.encoded_texts
            ]
        
        self.encoded_texts = [
            encoded_text + [pad_token_id]*(self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]['Label']
        
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
    
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        max_length = 0
        for text in self.encoded_texts:
            max_length = max(len(text), max_length)
        return max_length
    
