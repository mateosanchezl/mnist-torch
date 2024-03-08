### IMPORTS
# PyTorch 
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToTensor
# Helper functions
from ml_helper_functions import accuracy_fn, print_train_time, plot_pred_vs_true_grid
from timeit import default_timer as timer
# Matplotlib
import matplotlib.pyplot as plt


def run(BATCH_SIZE: int, LEARNING_RATE: float, EPOCHS: int, path: str, OPTIMISER="SGD"):
    """Train a neural network to classify the MNIST Dataset. 

    Args:
        BATCH_SIZE (int): Training batch size
        LEARNING_RATE (float): Learning rate for optimiser
        EPOCHS (int): Epochs of training over the whole training dataset
        path (str): Desired path for dataset download
        OPTIMISER (str, optional): Choice of optimiser. Defaults to "SGD".

    Raises:
        Exception: Invalid optimiser

    Returns:
        Model performance, prediction plot
    """   
    print("Getting data...")
    ### GETTING DATA
    train_data = MNIST(
    root=path,
    train=True,
    transform=ToTensor(),
    download=True
    )

    test_data = MNIST(
        root=path,
        train=False,
        transform=ToTensor(),
        download=True
    )
    
    ### DATALOADER
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False) # No shuffle for consistent evaluation
    
    # DEVICE AGNOSTIC CODE
    device = "cpu"
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    
    class MNISTv2(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Flatten(), # to flatten image
                nn.Linear(784, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 10),
            )
        
        def forward(self, x):
            return self.layers(x)
        
    model_0 = MNISTv2()

    loss_fn = nn.CrossEntropyLoss()
    if OPTIMISER.lower() == "sgd":   
        optim = torch.optim.SGD(params=model_0.parameters(), lr=LEARNING_RATE)
    elif OPTIMISER.lower() == "adam":
        optim = torch.optim.Adam(params=model_0.parameters(), lr=LEARNING_RATE)
    else:
        raise Exception("Optimiser invalid. Only 'SGD' and 'Adam' available.")

    
    # Using tqdm for progress bar
    from tqdm.auto import tqdm

    epochs = EPOCHS

    train_time_start = timer()
    ### Looping through all batches
    print("Training...")
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n -----")
        
        ### Training
        train_loss = 0
        
        ### Looping per batch
        for batch, (X, y) in enumerate(train_dataloader):
            model_0 = model_0.to(device)
            X = X.to(device)
            y = y.to(device)
            
            
            model_0.train()
            
            # 1. Forward
            y_pred = model_0(X)
            
            # 2. Calculate loss (per batch)
            loss = loss_fn(y_pred, y)
            train_loss += loss # Accumulate train loss
            
            # 3. Zero grad
            optim.zero_grad()
            
            # 4. Backprop
            loss.backward()
            
            # 5. Step
            optim.step()
            
            
            if batch % 400 == 0:
                print(f"Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")
            
        # Average train loss
        train_loss /= len(train_dataloader)
        
        ### Testing loop
        test_loss, test_acc = 0, 0
        model_0.eval()
        
        with torch.inference_mode():
            for X_test, y_test in test_dataloader:
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                # 1. Foward 
                test_pred = model_0(X_test)
                
                # 2. Calculate loss (accumulatively)
                test_loss += loss_fn(test_pred, y_test)
                
                # 3. Calculate accuracy
                test_acc += accuracy_fn(y_test, test_pred.argmax(dim=1)) # Compare labels to labels
            
            # Calculate test loss average per batch
            test_loss /= len(test_dataloader)
            
            # Calculate test acc average per batch
            test_acc /= len(test_dataloader)
            
        print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:4f} | Test Accuracy: {test_acc:4f}")

    train_time_end = timer()

    print(f"\n### MODEL PERFORMANCE SUMMARY ### \n\nFinal Train Loss: {train_loss:4f} \nFinal Test Loss: {test_loss:4f} \nFinal Test Accuracy: {test_acc:4f}")                
    print_train_time(train_time_start, train_time_end, device=device)        