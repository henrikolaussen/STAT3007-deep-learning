import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        device = x.device
        self.encoding = self.encoding.to(device)
        return x + self.encoding[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, num_features, d_model, nhead, num_layers, output_size, max_len=5000):
        super(TransformerModel, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder = nn.Linear(num_features, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, batch_first=True)
        self.decoder = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        x = self.encoder(x)  # Project input to model dimension
        x = self.positional_encoding(x) # Add positional encoding
        x = self.transformer(x, x)
        x = self.decoder(x[:, -1, :])  
        return x

class GRU_ForeCastModel(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers, output_size, device):
        super(GRU_ForeCastModel, self).__init__()
        self.n_layers = n_layers
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device

        self.gru = nn.GRU(n_features,
                          hidden_size=hidden_size,
                          num_layers=n_layers, 
                          batch_first=True)
        
        self.fc1 = nn.Linear(n_features * hidden_size, output_size)

    def forward(self, x, unbatched = False):
        batch_size = x.size(0)
        if unbatched:
            h0 = torch.zeros(self.n_layers, self.hidden_size).to(self.device)
        else:
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.gru(x, h0)
        if unbatched:
            return self.fc1(out[-1,:])
        else:
            return self.fc1(out[:,-1,:])


class CNN1D_ForeCastModel(nn.Module):
    def __init__(self, n_features, sequence_length, output_size):
        super(CNN1D_ForeCastModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=16, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (sequence_length // 4), 50) 
        self.fc2 = nn.Linear(50, output_size)

    def forward(self, x):
        x = x.permute(0,2,1) #CNN expects (batch_size, n_features, sequence_length)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x