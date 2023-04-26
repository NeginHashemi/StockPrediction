import torch
import pandas as pd
from stock_env import read_csv


class StockDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Divar. """

    def __init__(self, tokenizer, xs, label_list=None, max_len=128):
        self.xs = xs
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {label: i for i, label in enumerate(label_list)} if isinstance(label_list, list) else {}
    
    def __len__(self):
        return len(self.xs)

    def __getitem__(self, item):
        x = self.xs[item]

        target = self.label_map.get(x['cat1'])

        desc_encoding = self.tokenizer.encode_plus(
            x['desc'],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt')        
        
        title_encoding = self.tokenizer.encode_plus(
            x['title'],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt')
        
        inputs = {
            'desc_input_ids': desc_encoding['input_ids'].flatten(),
            'desc_attention_mask': desc_encoding['attention_mask'].flatten(),
            'desc_token_type_ids': desc_encoding['token_type_ids'].flatten(),

            'title_input_ids': title_encoding['input_ids'].flatten(),
            'title_attention_mask': title_encoding['attention_mask'].flatten(),
            'title_token_type_ids': title_encoding['token_type_ids'].flatten(),
        }

        inputs['targets'] = torch.tensor(target, dtype=torch.long)
        
        return inputs


def StockLoader(stock_filepaths, column_names, batch_size, w, split=[0.8, 0.1]):
    stock_info = [read_csv(p) for p in stock_filepaths]

    train_pd = pd.concat([df[column_names].iloc[0:int(len(df) * split[0])] for df in stock_info], ignore_index=True)
    val_pd = pd.concat([df[column_names].iloc[int(len(df) * split[0]):int(len(df) * split[1])] for df in stock_info], ignore_index=True)
    test_pd = pd.concat([df[column_names].iloc[int(len(df) * split[1]):] for df in stock_info], ignore_index=True)

    train_dataset = StockDataset(train_pd, w)
    val_dataset = StockDataset(val_pd, w)
    test_dataset = StockDataset(test_pd, w)

    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dl, val_dl, test_dl
