from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from cnn_model import CNN

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 100
OPTIMIZER_DEFAULT = 'ADAM'

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    _, pred = torch.max(predictions, 1)
    correct = (pred == targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy

def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root=FLAGS.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=FLAGS.data_dir, train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = CNN(n_channels=3, n_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
    
    train_losses = []
    test_accuracies = []
    
    for epoch in range(FLAGS.max_steps):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        train_losses.append(running_loss / len(train_loader))
        
        if (epoch + 1) % FLAGS.eval_freq == 0:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    correct += (outputs.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            accuracy = correct / total
            test_accuracies.append(accuracy)
            print(f"Epoch [{epoch+1}/{FLAGS.max_steps}], Loss: {running_loss / len(train_loader):.4f}, Test Accuracy: {accuracy:.4f}")
    
    np.save('train_losses.npy', np.array(train_losses))
    np.save('test_accuracies.npy', np.array(test_accuracies))



def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = '../Part1/data/',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()