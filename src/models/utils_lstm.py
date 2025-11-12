import torch
from torch import nn

class VariationalLSTM(nn.Module):
    """
    Variational LSTM layer in Pytorch
    """
    def __init__(self, input_size, hidden_size, num_layer=1, dropout_rate=0.0):
        super().__init__()

        self.lstm_layers = [nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)]
        if num_layer > 1:
            self.lstm_layers += [nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
                                 for _ in range(num_layer - 1)]
        self.lstm_layers = nn.ModuleList(self.lstm_layers)

        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

    def forward(self, x, init_states=None):
        for lstm_cell in self.lstm_layers:

            # Customised LSTM-cell for variational LSTM dropout (Tensorflow-like implementation)
            if init_states is None:  # Encoder - init states are zeros
                hx, cx = torch.zeros((x.shape[0], self.hidden_size)).type_as(x), \
                    torch.zeros((x.shape[0], self.hidden_size)).type_as(x)
            else:  # Decoder init states are br of encoder
                hx, cx = init_states, init_states

            # Variational dropout - sampled once per batch
            out_dropout = torch.bernoulli(hx.data.new(hx.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
            h_dropout = torch.bernoulli(hx.data.new(hx.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
            c_dropout = torch.bernoulli(cx.data.new(cx.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)

            output = []
            for t in range(x.shape[1]):
                hx, cx = lstm_cell(x[:, t, :], (hx, cx))
                if lstm_cell.training:
                    out = hx * out_dropout
                    hx, cx = hx * h_dropout, cx * c_dropout
                else:
                    out = hx
                output.append(out)

            x = torch.stack(output, dim=1)

        return x
    

    def encode(self, prev_treatments, vitals, prev_outputs, static_features, active_entries, init_states=None):
        static_features = static_features.unsqueeze(1).expand(-1, prev_treatments.size(1), -1)
        x = torch.cat([static_features, vitals, prev_treatments, prev_outputs.unsqueeze(-1)], dim=-1)
        return self.forward(x, init_states=init_states)

    def single_step(self, x, hidden_states):
        """
        Process one time step through all layers using the provided hidden states.

        Args:
            x: Input tensor of shape (b, dx), where only the first time step (t=0) is processed.
            hidden_states: List of (hx, cx) tuples, one per layer, each of shape (b, hidden_size).
            dropout_masks: Optional list of (out_dropout, h_dropout, c_dropout) tuples, one per layer,
                        each of shape (b, hidden_size). If None and training, no dropout is applied.

        Returns:
            output: Output tensor of shape (b, hidden_size) from the last layer.
            new_hidden_states: List of updated (hx, cx) tuples for each layer.
        """
        assert len(hidden_states) == len(self.lstm_layers), "hidden_states must match the number of layers"
        assert hidden_states[0][0].shape[1] == self.hidden_size, "hidden_states must have the same hidden size as the layer"
        assert x.dim() == 2, "Input x must be 2D (b, dinput)"
        
        new_hidden_states = []
        # Take the first time step from x; shape becomes (b, dx)
        input = x

        # Process through each layer
        for l, lstm_cell in enumerate(self.lstm_layers):
            hx, cx = hidden_states[l]
            # Apply LSTMCell for one time step
            hx, cx = lstm_cell(input, (hx, cx))

            out_dropout = torch.bernoulli(hx.data.new(hx.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
            h_dropout = torch.bernoulli(hx.data.new(hx.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
            c_dropout = torch.bernoulli(cx.data.new(cx.data.size()).fill_(1 - self.dropout_rate)) / (1 - self.dropout_rate)
            
            # Apply variational dropout if training and masks are provided
            if self.training:
                out = hx * out_dropout
                hx = hx * h_dropout
                cx = cx * c_dropout
            else:
                out = hx
            
            # Store updated hidden states
            new_hidden_states.append((hx, cx))
            # Output becomes input to the next layer
            input = out

        return out, new_hidden_states