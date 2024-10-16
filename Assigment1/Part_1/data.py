import numpy as np
import pandas as pd

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


if __name__ == "__main__":

    train_data, test_data = generate_data()
    print(train_data.head())
    print(test_data.head())
