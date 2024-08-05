# Should be a single file for running training runs of the LSTM models and log the results to wandb
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from stat3007_timeseries_forecasting.dataset import SequenceDataset
from stat3007_timeseries_forecasting.lstm.models import *
from stat3007_timeseries_forecasting.GRUand1Dconv.models import *
from stat3007_timeseries_forecasting.utils import (
    get_data_dir,
    get_sequences_multiple_tickers
)

# Determine the device based on availability
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

input_dim = 1
hidden_dim = 4096
num_layers = 1
output_dim = 5

prediction_horizon = output_dim
batch_size = 256
test_batch_size = 16

learning_rate = 0.01
input_sequence_length = 100
num_epochs = 50

model_type = "LSTM"

wandb.init(
    project="stat3007", entity="lars-ostberg-moan",
    name=f"{model_type}_b{batch_size}_num_epochs{num_epochs}_hidden{hidden_dim}"
)

wandb.config.update(
    {"learning_rate": learning_rate, "batch_size": batch_size, 
     "num_epochs": num_epochs,
    "hidden_dim": hidden_dim, 
    "num_layers": num_layers,
    "input_sequence_length": input_sequence_length,
    "prediction_horizon": prediction_horizon}
)



# Data prep
data = pd.read_csv(get_data_dir() + "energy_sector_dataset.csv")
train_sequences, test_sequences = get_sequences_multiple_tickers(
    data, input_sequence_length, train_percentage=0.8
)

#model = TransformerModel(1, d_model=hidden_dim, nhead=4, num_layers=1, output_size=5).to(device)

#model = GRU_ForeCastModel(1, hidden_size=hidden_dim, n_layers=1, output_size=5, device=device).to(device)

#model = CNN1D_ForeCastModel(n_features=1, sequence_length=input_sequence_length, output_size=5).to(device)

model = ForeCastModel(input_dim, hidden_dim, num_layers, output_dim, device).to(device)

#model = RNNForeCast(input_size=1, hidden_size=hidden_dim, num_layers=num_layers).to(device)

train_dataset = SequenceDataset(train_sequences)
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

test_dataset = SequenceDataset(test_sequences)
test_dataloader = DataLoader(test_dataset, test_batch_size, drop_last=True)

criterion = torch.nn.MSELoss(reduction="mean")
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    epoch_train_running_loss = 0.0
    model.train()
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        x = batch[:, :-prediction_horizon].to(device)
        x = x.unsqueeze(dim=-1)  # Add singleton dimension so the model can process it

        targets = batch[:, -prediction_horizon:].to(device)  # Last n datapoints is reserved for the target

        targets = targets.squeeze()


        outputs = model(x)  # Forward pass
        outputs = outputs.squeeze()  # [batch_size, 1] -> [batch_size]
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
        wandb.log({'train_loss': loss.item() / x.shape[0]})  
        epoch_train_running_loss += loss.item() / x.shape[0]

    epoch_train_loss = epoch_train_running_loss / len(train_dataloader)
    wandb.log({'train_epoch_loss': epoch_train_loss})

    #See how well it generalizes to new data
    model.eval()
    epoch_test_running_loss = 0.0
    with torch.no_grad():
        test_loss_running = 0
        for i, batch in enumerate(test_dataloader):
            print(i)
            x = batch[:, :-prediction_horizon].to(device)
            x = x.unsqueeze(dim=-1)  # Add singleton dimension so the model can process it

            targets = batch[:, -prediction_horizon:].to(device)  # Last n datapoints is reserved for the target
            targets = targets.squeeze()

            outputs = model(x)  # Forward pass
            outputs = outputs.squeeze()  # [batch_size, 1] -> [batch_size]

            loss = criterion(outputs, targets)
            epoch_test_running_loss += loss.item() / x.shape[0]

        epoch_test_loss = epoch_test_running_loss / len(test_dataloader)
        wandb.log({'test_epoch_loss': epoch_test_loss})


torch.save(model, get_data_dir() + f"weights/{model_type}_b{batch_size}_num_epochs{num_epochs}_hidden{hidden_dim}.pth")
wandb.finish()