from torch import nn


class MLP(nn.Module):
    """
    Regression approximation via 3-FC NN layers.
    The network input features are one-dimensional as well as the output features.
    The network hidden sizes are 128.
    Activations are ReLU
    """
    def __init__(self, D_in = 3, H = 128, D_out = 2):
        super().__init__()
        # --- Your code here
        self.linear1 = nn.Linear(D_in, H)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(H, H)  # Added second hidden layer
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(H, D_out)  # Output layer
        # ---

    def forward(self, x):
        """
        :param x: Tensor of size (N, 3)
        :return: y_hat: Tensor of size (N, 2)
        """
        # --- Your code here
        y_hat = self.linear1(x)
        y_hat = self.act1(y_hat)
        y_hat = self.linear2(y_hat)
        y_hat = self.act2(y_hat)
        y_hat = self.linear3(y_hat)
        
        # ---
        return y_hat
