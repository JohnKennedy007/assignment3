import torch
from torch import nn

def create_linear_regression_model(input_size, output_size):
    """
    Create a linear regression model with the given input and output sizes.
    """
    model = nn.Linear(input_size, output_size)
    return model

def train_iteration(X, y, model, loss_fn, optimizer):
    # Compute prediction and loss
    pred = model(X)
    loss = loss_fn(pred, y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def fit_regression_model(X, y):
    """
    Train the model for the given number of epochs.
    """
    learning_rate = 0.01 # Pick a better learning rate
    num_epochs = 10000 # Pick a better number of epochs
    input_features = X.shape[1] # extract the number of features from the input `shape` of X
    output_features = y.shape[1] # extract the number of features from the output `shape` of y
    model = create_linear_regression_model(input_features, output_features)
    
    loss_fn = nn.MSELoss() # Use mean squared error loss, like in class

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    previos_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        loss = train_iteration(X, y, model, loss_fn, optimizer)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: loss = {loss.item()}")

        # Stopping condition
        if abs(previos_loss - loss.item()) < 1e-6:
            print(f"Training stopped at epoch {epoch} due to minimal change in loss")
            break
        previos_loss = loss.item()

    return model, loss
