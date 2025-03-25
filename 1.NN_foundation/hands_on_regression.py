import torch
from typing import List, Tuple
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm


##############################################################################
# HANDS ON REGRESSION


def polynomial_basis_functions(xs: Tensor, d: int) -> Tensor:
    """
    Extends the input array to a series of polynomial basis functions of it.
    Args:
        xs: torch.Tensor (N, num_feats)
        d: Integer representing the degree of the polynomial basis functions
    Returns:
        Xs: torch.Tensor of shape (N, d*num_feats) containing the basis functions for the
        i.e. [1, x, x**2, x**3,...., x**d]
    """
    # --- Your code here
    N, _ = xs.shape
    Xs = []
    # Xs.append(xs)
    for deg in range(0, d + 1):
        Xs.append(xs**deg)
    Xs = torch.cat(Xs, dim = 1)
    # ---
    return Xs


def compute_least_squares_solution(Xs: Tensor, ys: Tensor) -> Tensor:
    """
    Compute the Least Squares solution that minimizes the MSE(Xs@coeffs, ys)
    Args:
        Xs: torch.Tensor shape (N,m)
        ys: Torch.Tensor of shape (N,1)
    Returns:
        coeffs: torch.Tensor of shape (m,) containing the optimal coefficients
    
    NOTE: You may need to compute the inverse of a matrix. Typically, computing 
    matrix inverses are a costly operation. Instead, given a linear system Ax = b,
    the solution can be computed much more efficient as x = torch.linalg.solve(A, b)
    """
    # coeffs = torch.zeros((Xs.shape[1], 1), requires_grad=True)
    # lr = 0.01
    # iter_times = 1000
    # for _ in range(iter_times):
    #     y_pred = Xs @ coeffs
    #     mse = torch.mean(torch.square(ys - y_pred))
    #     mse.backward()

    #     with torch.no_grad():
    #         coeffs -= lr * coeffs.grad
    #         coeffs.grad.zero_()
    # return coeffs.detach().squeeze()
        # Compute A = X^T X and b = X^T y
    A = Xs.T @ Xs  # (m, m)
    b = Xs.T @ ys  # (m, 1)

    # Solve Ax = b directly
    coeffs = torch.linalg.solve(A, b)  # (m, 1)

    # Return the coefficients as a 1D tensor
    return coeffs.squeeze()


def get_normalization_constants(Xs: Tensor) -> Tuple:
    # --- Your code here
    mean_i = torch.mean(Xs, dim=0)  # Compute mean per feature
    std_i = torch.std(Xs, dim=0)  # Compute std per feature
    # ---
    return mean_i, std_i


def normalize_tensor(Xs: Tensor, mean_i: Tensor, std_i: Tensor) -> Tensor:
    """
    Normalize the given tensor Xs
    :param Xs: torch.Tensor of shape (batch_size, num_features)
    :return: Normalized version of Xs
    """
    # --- Your code here
    Xs_norm = (Xs - mean_i)/std_i
    # ---
    Xs_norm = torch.nan_to_num(Xs_norm, nan=0.0) # avoid NaNs.
    return Xs_norm


def denormalize_tensor(Xs_norm: Tensor, mean_i: Tensor, std_i: Tensor) -> Tensor:
    """
        Normalize the given tensor Xs
        :param Xs: torch.Tensor of shape (batch_size, num_features)
        :return: Normalized version of Xs
        """
    # --- Your code here
    Xs_denorm = Xs_norm * std_i + mean_i
    # ---
    return Xs_denorm


class LinearRegressor(nn.Module):
    """
    Linear regression implemented as a neural network.
    The learnable coefficients can be easily implemented via linear layers without bias.
    The network regression output is one-dimensional.

    """
    def __init__(self, num_in_feats):
        super().__init__()
        self.num_in_feats = num_in_feats # number of regression input features
        # Define trainable
        # self.coeffs = torch.tensor(num_in_feats, requires_grad=True) # TODO: Override with the learnable regression coefficients
        # --- Your code here
        self.linear = torch.nn.Linear(num_in_feats, 1, bias=False)
        # ---

    def forward(self, x):
        """
        :param x: Tensor of size (N, num_in_feats)
        :return: y_hat: Tensor of size (N, 1)
        """
        # --- Your code here
        y_hat = self.linear(x)
        # ---
        return y_hat

    def get_coeffs(self):
        return self.coeffs.weight.data


class GeneralNN(nn.Module):
    """
    Regression approximation via 3-FC NN layers.
    The network input features are one-dimensional as well as the output features.
    The network hidden sizes are 100 and 100.
    Activations are Tanh
    """
    def __init__(self, D_in = 1, H = 100, D_out = 1):
        super().__init__()
        # --- Your code here
        self.linear1 = nn.Linear(D_in, H)
        self.act1 = nn.Tanh()
        self.linear2 = nn.Linear(H, H)  # Second hidden layer (missing before)
        self.act2 = nn.Tanh()
        self.linear3 = nn.Linear(H, D_out)  # Output layer
        # ---

    def forward(self, x):
        """
        :param x: Tensor of size (N, 1)
        :return: y_hat: Tensor of size (N, 1)
        """
        # --- Your code here
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)  # Second activation function (previously missing)
        y_hat = self.linear3(x)
        # ---
        return y_hat


def train_step(model, train_loader, optimizer) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    model.train()
    train_loss = 0. # TODO: Modify the value
    # Initialize the train loop
    # --- Your code here
    loss = nn.MSELoss()
    # ---
    for batch_idx, (data, target) in enumerate(train_loader):
        # --- Your code here
        optimizer.zero_grad()
        y_hat = model(data)  
        mse = loss(y_hat, target)
        mse.backward()
        optimizer.step()
        # ---
        train_loss += mse.item()
    return train_loss/len(train_loader)


def val_step(model, val_loader) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    model.eval()
    val_loss = 0. # TODO: Modify the value
    # Initialize the validation loop
    # --- Your code here
    loss = nn.MSELoss()
    # ---
    for batch_idx, (data, target) in enumerate(val_loader):
        # --- Your code here
        y_hat = model(data)
        loss = loss(y_hat, target)
        # ---
        val_loss += loss.item()
    return val_loss/len(val_loader)


def train_model_1(model, train_dataloader, val_dataloader, num_epochs=100, lr=1e-3):
    """
    Trains the given model for `num_epochs` epochs. Use SGD as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """
    # Initialize the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr)
    # --- Your code here
    # model = LinearRegressor()
    # ---
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
        # --- Your code here
        train_loss_i = train_step(model, train_dataloader, optimizer)
        val_loss_i = val_step(model, val_dataloader)
        # ---
        pbar.set_description(f'Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
        if epoch_i == 50 and train_loss_i > 0.025:
            break
        if epoch_i == 200 and train_loss_i > 0.015:
            break
        if epoch_i == 750 and train_loss_i > 0.01:
            break
    
    return train_losses, val_losses


def train_model(model, train_dataloader, val_dataloader, num_epochs=100, lr=1e-3):
    """
    Trains the given model for `num_epochs` epochs. Use SGD as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """
    # Initialize the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr)
    # --- Your code here
    # model = LinearRegressor()
    # ---
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
        # --- Your code here
        train_loss_i = train_step(model, train_dataloader, optimizer)
        val_loss_i = val_step(model, val_dataloader)
        # ---
        pbar.set_description(f'Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
        # if epoch_i == 50 and train_loss_i > 0.025:
        #     break
        # if epoch_i == 200 and train_loss_i > 0.015:
        #     break
        # if epoch_i == 750 and train_loss_i > 0.01:
        #     break
    
    return train_losses, val_losses


