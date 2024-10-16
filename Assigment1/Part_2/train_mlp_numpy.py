import csv
import argparse
import numpy as np
import pandas as pd
from mlp_numpy import MLP  
from sklearn.datasets import make_moons
from modules import CrossEntropy, Linear

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500 # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10

def generate_data():
    data, labels = make_moons(n_samples=1000, noise=0.1, random_state=42)
    
    df = pd.DataFrame(data, columns=['x', 'y'])
    df['label'] = labels

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_data = df.iloc[:800]
    test_data = df.iloc[800:]

    return train_data, test_data

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # TODO: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets
    predicted_classes = np.argmax(predictions, axis=1)
    correct_predictions = np.sum(predicted_classes == targets)
    return (correct_predictions / len(targets)) * 100

def train(dnn_hidden_units, learning_rate, max_steps, eval_freq):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        NOTE: Add necessary arguments such as the data, your model...
    """
    # TODO: Load your data here
    train_data, test_data = generate_data()
    
    x_train = train_data[['x', 'y']].values
    y_train = train_data['label'].values
    x_test = test_data[['x', 'y']].values
    y_test = test_data['label'].values
    
    # TODO: Initialize your MLP model and loss function (CrossEntropy) here

    mlp = MLP(n_inputs=2, n_hidden=[int(h) for h in dnn_hidden_units.split(',')], n_classes=2)
    cross_entropy = CrossEntropy()

    # Open a CSV file to log training progress
    with open('training_log.csv', mode='w') as log_file:
        log_writer = csv.writer(log_file)
        # Write header
        log_writer.writerow(['Step', 'Train Loss', 'Train Accuracy', 'Test Loss', 'Test Accuracy'])
        
        for step in range(max_steps):
            # TODO: Implement the training loop
            # 1. Forward pass
            # 2. Compute loss
            # 3. Backward pass (compute gradients)
            # 4. Update weights
            logits = mlp.forward(x_train)
            loss = cross_entropy.forward(logits, y_train)
            acc = accuracy(logits, y_train)
            dout = cross_entropy.backward(logits, y_train)
            mlp.backward(dout)
        
            # Update weights using gradients
            for layer in mlp.layers:
                if isinstance(layer, Linear):
                    layer.params['weight'] -= learning_rate * layer.grads['weight']
                    layer.params['bias'] -= learning_rate * layer.grads['bias']
        
            if step % eval_freq == 0 or step == max_steps - 1:
                # TODO: Evaluate the model on the test set
                # 1. Forward pass on the test set
                # 2. Compute loss and accuracy
                test_logits = mlp.forward(x_test)
                test_loss = cross_entropy.forward(test_logits, y_test)
                test_accuracy = accuracy(test_logits, y_test)

                # Log training and testing loss and accuracy
                log_writer.writerow([step, loss, acc, test_loss, test_accuracy])
                print(f"Step: {step}, Loss: {loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    
    print("Training complete!")

def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS = parser.parse_known_args()[0]
    
    train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

if __name__ == '__main__':
    main()
