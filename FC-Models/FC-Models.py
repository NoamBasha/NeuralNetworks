import sys

import matplotlib.pyplot as plt

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class ModelClassicReLU(nn.Module):
    def __init__(self, image_size):
        super(ModelClassicReLU, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


class ModelDropoutReLU(nn.Module):
    def __init__(self, image_size, dropout):
        super(ModelDropoutReLU, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 100)
        self.do0 = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(100, 50)
        self.do1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.do0(x)
        x = F.relu(self.fc1(x))
        x = self.do1(x)
        return F.log_softmax(self.fc2(x), dim=1)


class ModelFiveReLU(nn.Module):
    def __init__(self, image_size):
        super(ModelFiveReLU, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return F.log_softmax(self.fc6(x), dim=1)


class ModelFiveSigmoid(nn.Module):
    def __init__(self, image_size):
        super(ModelFiveSigmoid, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)
        self.fc6 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.sigmoid(self.fc0(x))
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        return F.log_softmax(self.fc6(x), dim=1)


class ModelBatchReLU(nn.Module):
    def __init__(self, image_size, optimizer=optim.Adam):
        super(ModelBatchReLU, self).__init__()
        self.image_size = image_size
        self.fc0 = nn.Linear(image_size, 256)
        self.ba0 = nn.BatchNorm1d(num_features=256)
        self.fc1 = nn.Linear(256, 64)
        self.ba1 = nn.BatchNorm1d(num_features=64)
        self.fc2 = nn.Linear(64, 10)
        self.ba2 = nn.BatchNorm1d(num_features=10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.fc0(x)
        x = F.relu(self.ba0(x))
        x = self.fc1(x)
        x = F.relu(self.ba1(x))
        x = self.fc2(x)
        return F.log_softmax(self.ba2(x), dim=1)


class MNISTNetwork:
    def __init__(self, network_architecture: nn.Module, train_loader, optimizer=optim.Adam, learning_rate=0.01,
                 epochs=15):
        self._epochs = epochs
        self.train_loader = train_loader
        self.network = network_architecture
        self.average_epoch_loss = []
        self.accuracy = []
        self.optimizer = optimizer(self.network.parameters(), lr=learning_rate)

    def train(self):
        for i in range(self._epochs):
            self.train_epoch()

    def train_epoch(self):
        epoch_loss = 0
        correct = 0
        self.network.train()
        for batch_idx, (sample, label) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.network.forward(sample)
            loss = F.nll_loss(output, label)
            epoch_loss += loss.item() * sample.size(0)
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).cpu().sum()
            loss.backward()
            self.optimizer.step()
        self.average_epoch_loss.append(epoch_loss / len(self.train_loader.dataset))
        self.accuracy.append(100 * correct / len(self.train_loader.dataset))

    def test(self, tests):
        self.network.eval()
        with torch.no_grad():
            return [self.network(test).max(1, keepdim=True)[1] for test in tests]

    def validate(self, labeled_tests):
        self.network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in labeled_tests:
                output = self.network(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).cpu().sum()
            test_accuracy = 100 * correct / len(labeled_tests.dataset)
            return test_loss / len(labeled_tests.dataset), test_accuracy

    def train_and_validate(self, labeled_tests):
        test_loss, test_accuracy = [], []
        for i in range(self._epochs):
            # Split data to train and validation
            self.train_epoch()
            epoch_loss, epoch_accuracy = self.validate(labeled_tests)
            test_loss.append(epoch_loss)
            test_accuracy.append(epoch_accuracy)
        self.plot_graphs(test_loss, test_accuracy)

    def plot_graphs(self, test_loss, test_accuracy):
        """
        Plot loss and accuracy graphs
        :return:
        """
        self.accuracy = np.array([acc.detach().numpy() for acc in self.accuracy])
        test_accuracy = np.array([acc.detach().numpy() for acc in test_accuracy])
        self.average_epoch_loss = np.array(self.average_epoch_loss)
        test_loss = np.array(test_loss)

        _, plots = plt.subplots(2, 1)
        range_reshaped = np.array(range(1, self._epochs + 1))
        range_reshaped.reshape(1, range_reshaped.shape[0])
        self.average_epoch_loss.reshape(1, self.average_epoch_loss.shape[0])
        self.accuracy.reshape(1, self.accuracy.shape[0])
        test_accuracy.reshape(1, test_accuracy.shape[0])
        test_loss.reshape(1, test_loss.shape[0])
        plots[0].plot(range_reshaped, self.average_epoch_loss, label="Train")
        plots[0].plot(range_reshaped, test_loss, label="Validation")
        plots[0].legend()
        plots[0].set_xlabel("Epoch")
        plots[0].set_ylabel("Average Loss")

        plots[1].plot(range_reshaped, self.accuracy, label="Train")
        plots[1].plot(range_reshaped, test_accuracy, label="Validation")
        plots[1].legend()
        plots[1].set_xlabel("Epoch")
        plots[1].set_ylabel("Accuracy")

        plt.show()


class MyDataset(Dataset):

    def __init__(self, X, y):
        self.x = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.n_samples = len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


def train_val_split(samples, labels, split_percentage):
    randomize = np.arange(len(samples))
    np.random.shuffle(randomize)
    samples = samples[randomize]
    labels = labels[randomize]
    split_index = int(split_percentage * len(samples))
    samples_train = samples[:split_index]
    samples_val = samples[split_index:]
    labels_train = labels[:split_index]
    labels_val = labels[split_index:]
    return [samples_train, labels_train, samples_val, labels_val]


def run_models(train_loader, val_loader, image_size):
    model_a = ModelClassicReLU(image_size)
    model_a_network = MNISTNetwork(model_a, train_loader, optimizer=optim.SGD)
    model_a_network.train_and_validate(val_loader)

    model_b = ModelClassicReLU(image_size)
    model_b_network = MNISTNetwork(model_b, train_loader, optimizer=optim.Adam)
    model_b_network.train_and_validate(val_loader)

    model_c = ModelDropoutReLU(image_size, dropout=0.1)
    model_c_network = MNISTNetwork(model_c, train_loader, optimizer=optim.Adam)
    model_c_network.train_and_validate(val_loader)

    model_d = ModelBatchReLU(image_size)
    model_d_network = MNISTNetwork(model_d, train_loader, optimizer=optim.Adam, learning_rate=0.001, epochs=7)
    model_d_network.train_and_validate(val_loader)

    model_e = ModelFiveReLU(image_size)
    model_e_network = MNISTNetwork(model_e, train_loader, optimizer=optim.Adam)
    model_e_network.train_and_validate(val_loader)

    model_f = ModelFiveSigmoid(image_size)
    model_f_network = MNISTNetwork(model_f, train_loader, optimizer=optim.Adam)
    model_f_network.train_and_validate(val_loader)


# Get data and normalize
train_x = np.loadtxt(sys.argv[1]).astype(float) / 255
train_y = np.loadtxt(sys.argv[2]).astype(int)
test_x = np.loadtxt(sys.argv[3]).astype(float) / 255

test_x = torch.from_numpy(test_x).float()

# Shuffle and split data
# samples_train, labels_train, samples_val, labels_val = train_val_split(train_x, train_y, 0.8)

# Create dataset
train_dataset = MyDataset(train_x, train_y)

# Create loader
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

image_size = 784
model_d = ModelBatchReLU(image_size)
model_d_network = MNISTNetwork(model_d, train_loader, optimizer=optim.Adam, learning_rate=0.001, epochs=7)
model_d_network.train()
predictions = model_d_network.test(test_x)
predictions = [x.item() for x in predictions]

# Writing the result to the output file
out_f = open(sys.argv[4], "w")
for i in range(len(test_x)):
    out_f.write(f"{predictions[i]}\n")
out_f.close()
