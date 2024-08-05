import torch
import torch.nn as nn

class RNNForeCast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNForeCast, self).__init__()
        self.rnn_layer = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.dense_layer = nn.Linear(hidden_size, 5)

    def forward(self, x):
        out = self.dense_layer(self.rnn_layer(x)[0][:,-1,:])
        return out
    


class ForeCastModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
        super(ForeCastModel, self).__init__() #Calls the constructor of the superclass nn.Module
        self.device = device

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input: torch.tensor, unbatched: bool = False):
        batch_size = input.shape[0]
        if unbatched:
            h0 = torch.zeros(self.num_layers, self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.num_layers, self.hidden_dim).requires_grad_()
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).requires_grad_()

        h0 = h0.to(self.device)
        c0 = c0.to(self.device)

        out, (hn, c_n) = self.lstm(input, (h0, c0))
        if unbatched:
            out = self.fc(out[-1, :])  # out[-1] will give the hidden state at the last time step
        else:
            out = self.fc(out[:, -1, :])  # out[:, -1, :] will give the hidden state at the last time step for each sequence
        return out