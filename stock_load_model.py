import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import ta
from ta.momentum import StochasticOscillator
import torch.optim as optim
import mlflow
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score, accuracy_score, cohen_kappa_score, matthews_corrcoef


path = 'vatejarat_30min.csv'
window_size = 100
future_prediction = 14

def read_csv(path):
    df = pd.read_csv(path)
    l = []
    for i in range(df.shape[0]-1, 0, -1):
        if abs((df.loc[i, 'open'] - df.loc[i-1, 'close']) / df.loc[i-1, 'close']) > 0.1:
            temp = df.loc[i, 'open'] - df.loc[i-1, 'close']
            print(i, '->', temp)
            l += [i]

    for i in l:
        temp = df.loc[i, 'open'] - df.loc[i-1, 'close']
        df.loc[:i-1, ['Price_first', 'Price_max', 'Price_min', 'Price_last', 'high', 'low', 'open', 'close']] += temp

    # print("    STEP 1")
    min_values = df[['Price_first', 'Price_max', 'Price_min', 'Price_last', 'high', 'low', 'open', 'close']].min()
    min = np.min(list(min_values))
    df.loc[:, ['Price_first', 'Price_max', 'Price_min', 'Price_last', 'high', 'low', 'open', 'close']] += -min + 100

    open = [0]
    close = [0]
    high = [0]
    low = [0]

    for i in range(1, df.shape[0]):
        open += [(df.loc[i, 'open'] - df.loc[i-1, 'open']) / df.loc[i-1, 'open']]
        close += [(df.loc[i, 'close'] - df.loc[i-1, 'close']) / df.loc[i-1, 'close']]
        high += [(df.loc[i, 'high'] - df.loc[i-1, 'high']) / df.loc[i-1, 'high']]
        low += [(df.loc[i, 'low'] - df.loc[i-1, 'low']) / df.loc[i-1, 'low']]

    df['open%'] = open
    df['close%'] = close
    df['high%'] = high
    df['low%'] = low

    # print("    STEP 2")


    df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    # Create stochastic oscillator with default parameters
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])

    # Add %K and %D lines to DataFrame
    df['%K'] = stoch.stoch()
    df['%D'] = stoch.stoch_signal()

    df['Williams_%R'] = ta.momentum.WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close']).williams_r()
    # calculate the accumulation distribution line using the ta library
    adl = ta.volume.AccDistIndexIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        volume=df['volume'],
    ).acc_dist_index()

    # print("    STEP 3")

    # add the ADL values to your DataFrame
    df['ADL'] = adl

    # Calculate CMF using ta
    df['cmf'] = ta.volume.ChaikinMoneyFlowIndicator(
        high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).chaikin_money_flow()

    # Calculate the OBV values using the ta library
    obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

    # Add the OBV values to your DataFrame as a new column
    df['obv'] = obv

    df['macd'] = ta.trend.MACD(df['close']).macd()

    indicator_cloud = ta.trend.IchimokuIndicator(high=df["high"], low=df["low"])
    df['ichimoku_base_line'] = indicator_cloud.ichimoku_base_line()
    df['ichimoku_conversion_line'] = indicator_cloud.ichimoku_conversion_line()
    df['ichimoku_a'] = indicator_cloud.ichimoku_a()
    df['ichimoku_b'] = indicator_cloud.ichimoku_b()


    df.dropna(inplace=True)

    df = df.reset_index(drop=True)

    columns_selection = ['open%', 'close%', 'high%', 'low%', 'RSI', '%K', '%D', 'Williams_%R', 'cmf']

    df_finals = df[columns_selection]

    X = df_finals.values.tolist()

    return X

def prepare_input(x):
    input_ids = []

    for i in range(len(x)-window_size):
        temp = []
        for j in range(window_size):
            temp += [torch.tensor(x[i+j], dtype=torch.float32)]
        input_ids.append(torch.stack(temp))

    return torch.tensor(torch.stack(input_ids, dim=0))


# define the model architecture
class IntegerTransformer3(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_heads, num_layers):
        super().__init__()
        self.conv = nn.Conv2d(kernel_size=(1, embedding_dim), out_channels=1, in_channels=1)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(num_embeddings, num_heads),
            num_layers
        )

        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)


    def forward(self, input_ids):
        embedded = self.conv(input_ids)
        output = self.transformer(embedded.squeeze(3))
        out = self.fc1(output.squeeze(1))
        out = self.relu(out)
        out = self.fc2(out).softmax(dim=1)
        return out

def test_model(x):  
    model = IntegerTransformer3(num_embeddings=window_size, embedding_dim=9, num_heads=4, num_layers=2)
    model.load_state_dict(torch.load('modelV1.pth', map_location=torch.device('cpu')))
    with torch.no_grad():
        output = model(x.unsqueeze(1)).numpy()
    return output

def test_csv(path):
    print('Reading file : ', path)
    X = read_csv(path)
    x = prepare_input(X)
    print('File loaded : ', path)
    output = test_model(x)
    print('Output generated', path)
    return output