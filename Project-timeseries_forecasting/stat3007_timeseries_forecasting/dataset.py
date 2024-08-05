import matplotlib.pyplot as plt
import pandas as pd
import torch
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import numpy as np

from stat3007_timeseries_forecasting.utils import get_data_dir


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]).float()



# Used to create a dataset for a single stock, in this case EUINOR
def create_equinor_dataset():
    eqnr = yf.Ticker("EQNR")
    data = eqnr.history(
        start="2001-06-18", end="2024-04-22", interval="1d"
    )  # Dates are chosen so that everyone will get the same dataset
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data["Scaled_close"] = scaler.fit_transform(data[["Close"]])
    plt.plot(data["Close"])
    plt.show()
    data.to_csv(get_data_dir() + "eqnr_max_daily.csv")


# Creates a dataset for a whole sector based on tickers defined in the csv file. Hardcoded in this example
def create_energysector_dataset():
    tickers_df = pd.read_csv(get_data_dir() + "nasdaq_energy_sector.csv")
    tickers_list = list(tickers_df["Symbol"].values)

    scaler = MinMaxScaler(feature_range=(-1, 1))

    dataframes = []

    for ticker in tickers_list:
        # Download historical data as dataframe
        data = yf.download(ticker, start="1900-01-01", end="2024-05-05")
        data = data.dropna(subset="Close")
        data["Ticker"] = ticker
        data["Scaled_close"] = scaler.fit_transform(
            data[["Close"]]
        )  # Creates a scaled close column on a per company basis.
        dataframes.append(data)

    dataset = pd.concat(dataframes)
    dataset.to_csv(get_data_dir() + "energy_sector_dataset.csv")

#inspired by the video by Rami Khushaba: https://www.youtube.com/watch?v=5vOYgJ-80Bg&t=1454s 
def calculate_sample_entropy(time_series, m: int = 2, r: float = 0.2): #returns a float between 0 and 1 representing how "random" a timeseries is.
    N = len(time_series)
    r = np.std(time_series)
    A = 0
    B = 0

    for i in range(N - m): #all possible window staring points in the series
        template_im = time_series[i:i + m]
        template_im1 = time_series[i:i + m + 1]

        for j in range(N - m): #comparing to all other windows
            if i != j:
                template_jm = time_series[j:j + m]
                template_jm1 = time_series[j:j + m + 1]

                if np.max(np.abs(template_im - template_jm)) < r:
                    A += 1

                    if np.max(np.abs(template_im1 - template_jm1)) < r:
                        B += 1

    if A == 0 or B == 0:
        return float('inf')
    
    similarity_ratio = B/A
    return -np.log(similarity_ratio)
            
if __name__ == "__main__":
    create_equinor_dataset()
    create_energysector_dataset()
