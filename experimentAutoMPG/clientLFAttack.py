import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow import keras
import flwr as fl
from sklearn.metrics import r2_score  # Import r2_score
from keras.metrics import R2Score, MeanAbsolutePercentageError

# Define a command-line argument for the portion of data
parser = argparse.ArgumentParser()
parser.add_argument("client_id", type=int, help="Client ID (1, 2, 3, ...)")
args = parser.parse_args()

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
print('Tf-version: ' , tf.__version__)

# Use the value of args.client_id to split the data accordingly
client_id = args.client_id
print(f"Client {client_id}")

# Get the data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()
print('Column names: ' , column_names)
dataset = dataset.sample(frac=1, random_state=0)

if client_id in [1,5]:
    # Client 1 gets the first 100 rows
    dataset = dataset.iloc[:100]
elif client_id in [2,6]:
    # Client 2 gets rows 100 to 200
    dataset = dataset.iloc[100:200]
elif client_id in [3,7]:
    # Client 3 gets rows 200 to 300
    dataset = dataset.iloc[200:300]
elif client_id in [4,8]:
    # Client 3 gets rows 200 to 393
    dataset = dataset.iloc[300:395]

# Clean the data
dataset.isna().sum()
dataset = dataset.dropna()
print(dataset)
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Origin'], prefix='', prefix_sep='')
dataset.tail()
dataset['USA'].replace({True: 1, False: 2}, inplace=True)
dataset['Europe'].replace({True: 1, False: 2}, inplace=True)
dataset['Japan'].replace({True: 1, False: 2}, inplace=True)
print(dataset)

# Split the data into training and test sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
print(train_labels)
test_labels = test_features.pop('MPG')

# Normalization
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

# Define a deep neural network model
dnn_model = keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
dnn_model.compile(
    optimizer="Adam",
    loss="mae",
    metrics=[R2Score(name='r2_score'), MeanAbsolutePercentageError(name='mape')])
    
# Function to shuffle "MPG" values
def flip_labels(labels):
    flipped_labels = np.random.permutation(labels)
    return flipped_labels


    
class FlowerClient(fl.client.NumPyClient):
    print("An adversarial client is started")
    def get_parameters(self, config):
        return dnn_model.get_weights()
    
    def fit(self, parameters, config):            
        dnn_model.set_weights(parameters)
        
        print("Train labels before flipping:")
        train_labels_series = pd.Series(train_labels)
        train_labels_series.index = range(len(train_labels))
        print(train_labels)

        # Shuffle the "MPG" values for the training data
        adversarial_train_labels = flip_labels(train_labels)

        print("Train labels after flipping:")
        adversarial_train_labels_series = pd.Series(adversarial_train_labels)
        adversarial_train_labels_series.index = range(len(train_labels))
        print(adversarial_train_labels)

        dnn_model.fit(train_features,
                    adversarial_train_labels,
                    validation_split=0.2,
                    epochs=50,
                    batch_size=25,
                    verbose=0
        )
        return dnn_model.get_weights(), len(train_features), {}
    
    def evaluate(self, parameters, config):
        dnn_model.set_weights(parameters)
        loss, r2_score, mape = dnn_model.evaluate(test_features,test_labels, verbose=1)
        print(f"Loss: {loss}, r2_score: {r2_score}, mape {mape}")  
        # Make predictions on the test data
        test_predictions = dnn_model.predict(test_features).flatten()
        
        return loss, len(test_features), {"r2_score": r2_score, "mape": mape}
        
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
