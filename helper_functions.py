# Timing function
from timeit import default_timer as timer
from matplotlib import pyplot as plt
import torch
import numpy as np
import torchvision
import random

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def plot_predictions(
    train_data, train_labels, test_data, test_labels, predictions=None
):
    """
  Plots linear training data and test data and compares predictions.
  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


def show_image_grid(data: torchvision.datasets, rows: int, cols: int, figsize: tuple, label: bool=False):
    """Plots a grid of images of size rows * cols from an image PyTorch dataset, starting at a random index.

    Args:
        data (torchvision.datasets): any torchvision image dataset
        rows (int): number of rows in grid
        cols (int): number of cols in grid
    """       
    fig = plt.figure(figsize=figsize)

    for i in range(1, rows * cols + 1):
        random_idx = random.randint(1, (len(data) - (rows * cols)))
        image = data[random_idx][0]
        label = data[random_idx][1]
        fig.add_subplot(rows, cols, i)
        plt.imshow(image.squeeze())
        # if label:
        #     plt.title(class_labels[label])
        # else:
        #     plt.title("")
        plt.axis("off")
        
def plot_pred_vs_true_grid(test_data: torchvision.datasets.MNIST, model: torch.nn.Module, rows: int, cols: int, figsize: tuple):
    fig = plt.figure(figsize=figsize)
    
    for i in range(1, rows * cols + 1):
        random_idx = random.randint(1, len(test_data) - 1)  # Corrected indexing
        x, y_true = test_data[random_idx]  # Extract input and true label from dataset
        y_pred_label = model(x.unsqueeze(0)).argmax(dim=1).item()  # Predict with model
        fig.add_subplot(rows, cols, i)
        plt.imshow(x.squeeze(), cmap='gray')  # Added cmap='gray' to display MNIST images correctly
        plt.title(f"Pred: {y_pred_label}, True: {y_true}")
        plt.axis("off")    
    plt.show()
