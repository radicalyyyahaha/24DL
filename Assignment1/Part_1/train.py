from data import generate_data
from perceptron import Perceptron 
import numpy as np
import pandas as pd

np.random.seed(1337)

def generate_data():
    mean1 = [2, 3]      
    cov1 = [[1, 0], [0, 1]]  

    mean2 = [7, 8]  
    cov2 = [[1, 0], [0, 1]] 

    points1 = np.random.multivariate_normal(mean1, cov1, 100)
    points2 = np.random.multivariate_normal(mean2, cov2, 100)

    labels1 = np.zeros(100) 
    labels2 = np.ones(100)  

    data = np.vstack([points1, points2])
    labels = np.concatenate([labels1, labels2])

    df = pd.DataFrame(data, columns=['x', 'y'])
    df['label'] = labels

    df = df.sample(frac=1).reset_index(drop=True)

    train_data = pd.concat([df[df['label'] == 0].iloc[:80], df[df['label'] == 1].iloc[:80]])
    test_data = pd.concat([df[df['label'] == 0].iloc[80:], df[df['label'] == 1].iloc[80:]])

    train_data = train_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.sample(frac=1).reset_index(drop=True)

    return train_data, test_data

train_data, test_data = generate_data()

train_inputs = train_data[['x', 'y']].values
train_labels = train_data['label'].apply(lambda x: 1 if x == 1 else -1).values  # Convert labels to 1 and -1

test_inputs = test_data[['x', 'y']].values
test_labels = test_data['label'].apply(lambda x: 1 if x == 1 else -1).values  # Convert labels to 1 and -1

# Initialize and train the perceptron
perceptron = Perceptron(n_inputs=2, max_epochs=100, learning_rate=0.01)
perceptron.train(train_inputs, train_labels)

# Test the perceptron on the test data
test_predictions = np.array([perceptron.forward(x) for x in test_inputs])

# Compute the number of correct and incorrect predictions
correct_predictions = np.sum(test_predictions == test_labels)
incorrect_predictions = np.sum(test_predictions != test_labels)

print(f"Number of correct predictions: {correct_predictions}")
print(f"Number of incorrect predictions: {incorrect_predictions}")