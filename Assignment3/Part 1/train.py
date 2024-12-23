from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import PalindromeDataset
from lstm import LSTM
from utils import AverageMeter, accuracy

import matplotlib.pyplot as plt

def plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train(model, data_loader, optimizer, criterion, device, config):
    model.train()  
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        # Forward pass
        outputs = model(batch_inputs)  
        outputs = outputs[:, -1, :]

        loss = criterion(outputs, batch_targets)  
        acc = accuracy(outputs, batch_targets)  
        

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        optimizer.step()

        losses.update(loss.item(), batch_inputs.size(0))
        accuracies.update(acc, batch_inputs.size(0))

        if step % 10 == 0:
            print(f"[{step}/{len(data_loader)}] Loss: {losses.avg:.4f}, Accuracy: {accuracies.avg:.4f}")

    return losses.avg, accuracies.avg


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, config):
    # TODO set model to evaluation mode
    model.eval()
    losses = AverageMeter("Loss")
    accuracies = AverageMeter("Accuracy")

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        outputs = model(batch_inputs) 

        final_output = outputs[:, -1, :]  
        loss = criterion(final_output, batch_targets)
        acc = accuracy(final_output, batch_targets)

        losses.update(loss.item(), batch_inputs.size(0))
        accuracies.update(acc, batch_inputs.size(0))

        if step % 10 == 0:
            print(f'[{step}/{len(data_loader)}]', losses, accuracies)

    return losses.avg, accuracies.avg


def main(config):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Initialize the model that we are going to use
    model = LSTM(seq_length=config.input_length,
                 input_dim=config.input_dim, 
                 hidden_dim=config.num_hidden, 
                 output_dim=config.num_classes)
    model.to(device)


    # Initialize the dataset and data loader
    dataset = PalindromeDataset(
        input_length=config.input_length, 
        total_len=config.data_size,
        one_hot=True
    )  # fixme

    # Split dataset into train and validation sets
    train_size = int(config.portion_train * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # fixme

    # Create data loaders for training and validation
    train_dloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)  # fixme
    val_dloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)  # fixme

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()  # fixme
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate, alpha=0.99, eps=1e-08)  # fixme
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # fixme

    # Store metrics for plotting
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(config.max_epoch):
        print(f"Epoch {epoch + 1}/{config.max_epoch}")
        train_loss, train_acc = train(
            model, train_dloader, optimizer, criterion, device, config)

        val_loss, val_acc = evaluate(
            model, val_dloader, criterion, device, config)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{config.max_epoch} | "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f} | "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")
    
        scheduler.step()

    plot_metrics(train_losses, train_accuracies, val_losses, val_accuracies)
    
    print('Done training.')


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=5,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=10,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float,
                        default=0.001, help='Learning rate')
    parser.add_argument('--max_epoch', type=int,
                        default=5, help='Number of epochs to run for')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--data_size', type=int,
                        default=100000, help='Size of the total dataset')
    parser.add_argument('--portion_train', type=float, default=0.8,
                        help='Portion of the total dataset used for training')

    config = parser.parse_args()
    # Train the model
    main(config)
