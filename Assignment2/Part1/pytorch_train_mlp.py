from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from pytorch_mlp import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '256'
LEARNING_RATE_DEFAULT = 1e-3
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 10

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
    pred = torch.argmax(predictions, dim=1)
    true = torch.argmax(targets, dim=1)
    correct = (pred == true).sum().item()
    accuracy = correct / targets.size(0)

    return accuracy

def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    hidden_units = list(map(int, FLAGS.dnn_hidden_units.split(',')))

    np.random.seed(42)
    torch.manual_seed(42)
    x_train = torch.tensor(np.random.rand(1000, 10), dtype=torch.float32)  
    y_train = torch.tensor(np.eye(3)[np.random.randint(0, 2, 1000)], dtype=torch.float32)  
    x_test = torch.tensor(np.random.rand(200, 10), dtype=torch.float32)
    y_test = torch.tensor(np.eye(3)[np.random.randint(0, 2, 200)], dtype=torch.float32)

    model = MLP(n_inputs=10, n_hidden=hidden_units, n_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=FLAGS.learning_rate)

    for epoch in range(FLAGS.max_steps):
        outputs = model(x_train)
        loss = criterion(outputs, torch.argmax(y_train, dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % FLAGS.eval_freq == 0:
            with torch.no_grad():
                test_outputs = model(x_test)
                test_acc = accuracy(test_outputs, y_test)
                print(f"Epoch [{epoch}/{FLAGS.max_steps}], Loss: {loss.item():.4f}, Test Accuracy: {test_acc:.4f}")



def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()