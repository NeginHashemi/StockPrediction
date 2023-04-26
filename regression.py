import torch
import numpy as np
import pandas as pd
from stock_env import read_csv


class StockDataset(torch.utils.data.Dataset):
    """ Create a PyTorch dataset for Divar. """

    def __init__(self, data_pd, w):
        self.data_pd = data_pd
        self.w = w

    def __len__(self):
        return len(self.data_pd - 2 * self.w)

    def __getitem__(self, item):
        item = self.w + item
        start_candle = self.data_pd.iloc[item]
        end_candle = self.data_pd.iloc[min(item+self.w, len(self.data_pd))]
        y = (end_candle['close'] - start_candle['open']) / end_candle['close']
        return self.data_pd.iloc[max(item-self.w, 0):item].to_numpy(), y


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


class Regression():
    def __init__(self, model, train_dl, val_dl, test_dl, device=None, lr=1e-5) -> None:
        self.model = model.to(device)
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl

        self.lr = lr
        self.device = device

        self.criterion = torch.nn.MSELoss()    # mean-squared error for regression
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_acc = 0
        total_confidence = 0
        cnt = 0
        for x, y in iter(self.train_dl):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            total_confidence += abs(outputs.detach().cpu().sum())
            total_acc += (torch.sign(outputs) == torch.sign(y)).sum().item() / torch.numel(outputs)
            cnt += 1

        return total_loss / cnt, total_confidence / cnt, total_acc / cnt

    def test_epoch(self, dl):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            total_acc = 0
            total_confidence = 0
            cnt = 0
            for x, y in iter(dl):
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss += loss.item()
                total_confidence += abs(outputs.detach().cpu().sum())
                total_acc += (torch.sign(outputs) == torch.sign(y)).sum().item() / torch.numel(outputs)
                cnt += 1
            return total_loss / cnt, total_confidence / cnt, total_acc / cnt

    def train(self, num_epoch=100, val_turn=10):
        history_df = pd.DataFrame(columns = ['total_acc', 'total_confidence', 'total_acc', 'is_train'])
        for e in range(num_epoch):
            total_loss, total_confidence, total_acc = self.train_epoch()
            print("Epoch {} ==== Loss: {}, Confidence: {}, Accuracy: {}".format(e, total_loss, total_confidence, total_acc))
            history_df = history_df.append(dict(total_loss=total_loss, total_confidence=total_confidence, total_acc=total_acc, is_train=True), ignore_index=True)
            if e % val_turn == 0 and e > 0:
                total_acc, total_confidence, total_acc = self.test_epoch(self.val_dl)
                print("================= Eval Epoch {} ==== Loss: {}, Confidence: {}, Accuracy: {}".format(e, total_loss, total_confidence, total_acc))
                history_df = history_df.append(dict(total_loss=total_loss, total_confidence=total_confidence, total_acc=total_acc, is_train=False), ignore_index=True)
        return history_df
            
            




