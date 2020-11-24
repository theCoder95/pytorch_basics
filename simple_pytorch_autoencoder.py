import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import random
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

'''
preprocessing functions start
'''
def read_data(path):
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis='columns')
    return df

def train_without_rul(path):
    dependent_var = ['RemainingUsefulLife']
    index_columns_names = ["UnitNumber", "Cycle"]
    operational_settings_columns_names = ["OpSet" + str(i) for i in range(1, 4)]
    sensor_measure_columns_names = ["SensorMeasure" + str(i) for i in range(1, 22)]
    input_file_column_names = index_columns_names + operational_settings_columns_names + sensor_measure_columns_names
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis='columns')
    df.columns = input_file_column_names
    return df

def test_data(path):
    dependent_var = ['RemainingUsefulLife']
    index_columns_names = ["UnitNumber", "Cycle"]
    operational_settings_columns_names = ["OpSet" + str(i) for i in range(1, 4)]
    sensor_measure_columns_names = ["SensorMeasure" + str(i) for i in range(1, 22)]
    input_file_column_names = index_columns_names + operational_settings_columns_names + sensor_measure_columns_names
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.dropna(axis='columns')
    df.columns = input_file_column_names
    return df

def serialize(df):
    time_series = []
    for i in range(1, df['UnitNumber'][len(df['UnitNumber'])-1]):
        tmp = df.loc[df['UnitNumber'] == i]
        time_series.append(tmp)
    return time_series


def train_and_valid_split(x, valid_size):
    num_train = len(x)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    return train_idx, valid_idx

'''
preprocessing functions end
'''

'''
Setup of the neural network start
'''

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size, epochs, learning_rate, print_training_epochs=50):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.print_training_epochs = print_training_epochs

        self.enc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(hidden_size, latent_size)
        )

        self.dec = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),#nn.Tanh(), #nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def encode(self, x):
        # encoding step
        latent = self.enc(x)
        return latent

    def decode(self, z):
        # decoding step
        decoded = self.dec(z)
        return decoded

    def forward(self, x):
        # a foreward step through the complete network
        # includes in this case: encoding and decoding
        latent = self.encode(x)
        decoded = self.decode(latent)
        return decoded

    def fit(self, x):
        # define a optimizer and define a Loss criterion
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss() # reduction='mean'
        # suffle the input: optional
        random.shuffle(x)
        mse_per_epoch_train = []
        mse_per_epoch_valid = []
        for epoch in range(1, self.epochs + 1):
            train_loss_per_epoch = []
            valid_loss_per_epoch = []
            valid_size = 0.2
            train_idx, valid_idx= train_and_valid_split(x, valid_size)

            # training loop
            self.train()
            for j in train_idx:
                x_i = x[j]
                optimizer.zero_grad()
                decoded = self.forward(x_i)
                loss = criterion(decoded, x_i)
                train_loss_per_epoch.append(loss.item())
                # Backward pass
                loss.backward()
                #nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.05)
                optimizer.step()
            mse_per_epoch_train.append(np.array(train_loss_per_epoch).mean())

            # validation loop (belongs also to training phase)
            self.eval()
            for j in valid_idx:
                x_i = x[j]
                decoded = self.forward(x_i)
                loss = criterion(decoded, x_i)
                valid_loss_per_epoch.append(loss.item())
            mse_per_epoch_valid.append(np.array(valid_loss_per_epoch).mean())

            if epoch % self.print_training_epochs == 0:
                print('epoch : ' + str(epoch))
                print(f"loss_mean train : {np.array(train_loss_per_epoch).mean():.10f}")
                print(f"loss_mean valid : {np.array(valid_loss_per_epoch).mean():.10f}")
        return mse_per_epoch_train, mse_per_epoch_valid

'''
Setup of the neural network end
'''


'''
additional helpers for the experiment start
'''

def prepare_dataset(sequential_data):
    if type(sequential_data) == pd.DataFrame:
        data_in_numpy = np.array(sequential_data)
        data_in_tensor = torch.tensor(data_in_numpy, dtype=torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(2)
    elif type(sequential_data) == np.array:
        data_in_tensor = torch.tensor(sequential_data, dtype=torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(2)
    elif type(sequential_data) == list:
        data_in_tensor = torch.tensor(sequential_data, dtype=torch.float)
        unsqueezed_data = data_in_tensor.unsqueeze(2)

    seq_len = unsqueezed_data.shape[1]
    no_features = unsqueezed_data.shape[2]
    # shape[0] is the number of batches
    return unsqueezed_data, seq_len, no_features

def get_decoded_as_pandas(data):
    dec = []
    for current in data:
        df = pd.DataFrame(current)
        df = df.transpose()
        dec.append(df)
    df = pd.DataFrame(dec[0])
    for i, c in enumerate(dec):
        c.index = [i]
        if i > 0:
            df = pd.concat([df, c])
    return df

def normal_plot(isNormal, original, new, labels, title):
    if isNormal:
        plt.plot(original, label='test data')
        plt.plot(new, label='autoencoded data')
        plt.legend(['test data', 'autoencoded data'])
        plt.title('Sensor Number 20')
        plt.show()
    else:
        plt.semilogy(original, label=labels[0])
        plt.semilogy(new, label=labels[1])
        plt.legend([labels[0], labels[1]])
        plt.title(title)
        plt.show()

'''
additional helpers for the experiment end
'''


'''
experiment setup start
'''

def experiment_start():
    # use this line for optimization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the prepocessed data
    X_train = serialize(train_without_rul('./data/train_FD001.txt'))
    X_test = serialize(test_data('./data/test_FD001.txt'))
    y_rul = read_data('./data/RUL_FD001.txt')

    # define parameters for the neural network
    hidden_size = 20
    latent_size = 8
    epochs = 200
    learning_rate = 0.0001 # mostly between 0.001 and 0.0001

    refined_input_data, seq_len, no_features = prepare_dataset(X_train[0])

    # initialize the neural network
    model = AutoEncoder(no_features, hidden_size, latent_size, epochs, learning_rate).to(device)

    X_train_redefined = []
    for i in range(0, len(X_train)):
        refined_input_data, seq_len, no_features = prepare_dataset(X_train[i])
        X_train_redefined.append(refined_input_data)

    # train the model
    mse_per_epoch, mse_per_epoch_valid = model.fit(X_train_redefined)
    print('final loss: ', mse_per_epoch)

    # semilogy plot of the mse
    normal_plot(False, mse_per_epoch, mse_per_epoch_valid, ['mse per epoch (train)', 'mse per epoch (validation)'], 'MSE per epoch on training data')

    # prepare test data for input to the trained model
    refined_input_data, seq_len, no_features = prepare_dataset(X_test[0])

    # use the trained model with the test data (or any other data)
    # does encoding and decoding with the trained model
    embedded_points = model.encode(refined_input_data)
    decoded_points = model.decode(embedded_points)

    df = get_decoded_as_pandas(decoded_points.cpu().data)

    print('mse on test serie: ', mean_squared_error(X_test[0], df))

    # plot original test data and reconstructed data
    normal_plot(True, X_test[0]['SensorMeasure20'], df[24], ['test data', 'autoencoded data'],
                'Sensor Number 20')

'''
experiment setup end
'''

if __name__ == "__main__":
    experiment_start()