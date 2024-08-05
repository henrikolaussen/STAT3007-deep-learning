import math
import os
from math import sqrt
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


# Return two list of sequences, one for training and one for testing. Train should have overlap, test not. Each sequence should have lenght x
def get_sequences(
    df: pd.DataFrame,
    train_sequence_length: int,
    test_sequence_length: int,
    train_percentage: float,
) -> Tuple[list, list]:
    data = df["Scaled_close"].values

    train_length = math.floor(len(data) * train_percentage)

    # Rolling window with 1 in stepsize to generate a lot of training samples.
    train_sequences = []
    for index in range(train_length - train_sequence_length + 1):
        sequence = data[index : index + train_sequence_length]
        train_sequences.append(sequence)

    # Not using rolling window for test sequences. Every sequence comes after the other with no overlap.
    test_sequences = []
    for index in range(
        train_length, len(data) - test_sequence_length + 1, test_sequence_length
    ):
        sequence = data[index : index + test_sequence_length]
        test_sequences.append(sequence)

    return np.array(train_sequences), np.array(test_sequences)


# Takes in a main dataframe with all datapoints, splits it by ticker value and creates train and test sequences on a per ticker basis. Concatenates the result into train and test sequences
# Key difference between train sequences and test sequences is that train has overlap whereas test sequences are after one another with no overlap.
def get_sequences_multiple_tickers(
    df: pd.DataFrame,
    input_sequence_length: int,
    train_percentage: float,
    test_offset: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    sequence_length = input_sequence_length + test_offset

    subframes = df.groupby("Ticker")
    train_sequences = []
    test_sequences = []
    i = 0
    for _, subframe in subframes:
        subframe_data = subframe["Scaled_close"].values
        train_length = int(len(subframe_data) * train_percentage)

        train_sequences_ticker = []
        for index in range(train_length - sequence_length + 1):
            sequence = subframe_data[index : index + sequence_length]
            train_sequences_ticker.append(sequence)

        train_sequences.extend(train_sequences_ticker)

        test_sequences_ticker = []
        for index in range(
            train_length, len(subframe_data) - sequence_length + 1, sequence_length
        ):
            sequence = subframe_data[index : index + sequence_length]
            test_sequences_ticker.append(sequence)

        test_sequences.extend(test_sequences_ticker)

    train_sequences = np.array(train_sequences, dtype=object).astype(np.float32)
    test_sequences = np.array(test_sequences, dtype=object).astype(np.float32)

    return train_sequences, test_sequences


def train(model, optimizer, dataloader, num_epochs, loss_func, prediction_horizon: int):
    device = model.device
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = len(dataloader)
        for i, batch in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)
        ):
            optimizer.zero_grad()

            x = batch[:, :-prediction_horizon].to(
                device
            )  # All but the last n elements of each sequence
            targets = batch[:, -prediction_horizon:].to(
                device
            )  # Last n elements is targets
            targets = targets.squeeze()  # [batch_size, 1, 1] -> [batch_size]

            x = x.unsqueeze(
                dim=-1
            )  # Add singleton dimension so the model can process it

            outputs = model(x)  # Forward pass
            outputs = outputs.squeeze()  # [batch_size, 1] -> [batch_size]

            loss = loss_func(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / total_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.10f}")


def test_model(
    model, test_sequences, prediction_horizon: int = 5
):
    device = model.device
    model.eval()

    predictions = []
    with torch.no_grad():
        for i in range(len(test_sequences)):
            input_length = len(test_sequences[i]) - prediction_horizon
            input = test_sequences[i][0:input_length]
            target = test_sequences[i][input_length : input_length + prediction_horizon]

            input_tensor = torch.tensor(input).to(device)
            input_tensor = input_tensor.unsqueeze(
                dim=-1
            )  # Adds singleton dimension along last axis [seqlen 1]

            prediction = model(input_tensor, unbatched=True).cpu()

            predictions.append(prediction)

    return predictions


# Used for unrolling time for single step prediction models to get quasi multistep predictions
def predict_timeseries(
    model, input_sequence: np.ndarray, n_timesteps: int, accumulative: bool = False
) -> list:
    model.eval()
    device = model.device
    with torch.no_grad():
        output_sequence = []

        input_tensor = torch.tensor(input_sequence).to(
            device
        )  # Converts to float32 and to a tensor from ndarray
        input_tensor = input_tensor.unsqueeze(
            dim=-1
        )  # Adds singleton dimension along last axis [seqlen 1]
        for index in range(n_timesteps):
            output = model(input_tensor, unbatched=True)
            output_sequence.append(
                output.squeeze().cpu().item()
            )  # Item get's the raw value of the tensor

            if (
                accumulative
            ):  # Just ass the prediction without removing the first element.
                input_tensor = torch.cat((input_tensor, output.unsqueeze(0)), dim=0)
            else:  # Rolling horizon of fixed length
                input_tensor = torch.roll(
                    input_tensor, -1, dims=0
                )  # Shift input_tensor by one timestep backwards
                input_tensor[-1] = (
                    output  # Update the last value of the input tensor to the output of the model.
                )

        return np.array(output_sequence)


def plot_predictions(
    input_sequences,
    predictions,
    prediction_horizon: int = 5,
    sequences_per_plot: int = 3,
):
    plt.figure(figsize=(30, 6))  # Width is 12 inches, height is 6 inches
    for i, input_sequence in enumerate(input_sequences):
        x_array_offset = np.arange(
            len(input_sequence) - prediction_horizon, len(input_sequence)
        )  # Used for both target and predictions when plotting

        
        plt.plot(
            input_sequence[: len(input_sequence) - prediction_horizon],
            color="blue",
            label="Input data",
        )

        plt.plot(
            x_array_offset,
            input_sequence[len(input_sequence) - prediction_horizon :],
            color="green",
            label="Target data",
        )
        plt.plot(
            x_array_offset, predictions[i], color="orange", label="Prediction"
        )

        plt.xlabel("Time Steps")
        plt.ylabel("Close price (scaled)")
        plt.grid(True)

        if (i + 1) % sequences_per_plot == 0:
            plt.show()
            

def get_data_dir():
    load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    if data_dir is None:
        raise ValueError("DATA_DIR environment variable is not set in the .env file")
    return data_dir


def unroll_model(model, starting_seq, day):
    window = starting_seq.unsqueeze(0)
    preds = []
    model.eval()
    for i in range(day):
        pred = model(window)
        pred = pred.squeeze(0).squeeze(0)
        preds.append(pred.detach().numpy())
        window = torch.cat((window[:, 1:], torch.tensor([[pred]]).unsqueeze(0)), dim=1)

    return np.array(preds)


def unroll_all_seq(model, seqs, days):
    preds = []
    for seq in seqs:
        preds.append(unroll_model(model, seq, days))
    return np.array(preds)


def plot_model_preds(model, seqs, days):
    preds = unroll_all_seq(model, seqs, days)
    fig = plt.figure()
    seq_lengths = len(seqs[0].squeeze(1))

    x = np.arange(seq_lengths + days)
    for idx, pred in enumerate(preds):
        seq = seqs[idx].squeeze(1).detach().numpy()
        # Add an offset to each sequence to avoid overlapping plots
        pred_copy = pred + idx * 0.02
        seq_copy = seq + idx * 0.02
        # Plot the sequence and prediction
        plt.plot(x[:seq_lengths], seq_copy)
        plt.plot(x[seq_lengths:], pred_copy)
    plt.show()


# Test the loading of the DATA_DIR variable from .env
if __name__ == "__main__":
    data_dir = get_data_dir()
    print("Data Directory:", data_dir)
