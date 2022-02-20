import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from gcommand_dataset import GCommandLoader

import sys


class CNNNetwork(nn.Module):

    def __init__(self, train_loader, epochs, learning_rate, device):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Sequential(nn.Linear(2048, 850), nn.BatchNorm1d(num_features=850), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(850, 350), nn.BatchNorm1d(num_features=350), nn.ReLU())
        self.linear3 = nn.Sequential(nn.Linear(350, 200), nn.BatchNorm1d(num_features=200), nn.ReLU())
        self.linear4 = nn.Linear(200, 30)

        self.softmax = nn.Softmax(dim=1)
        optimizer = optim.Adam
        self.optimizer = optimizer(self.parameters(), lr=learning_rate)
        self.epochs = epochs
        self.train_loader = train_loader
        self.device = device
        self.to(device)

        self.average_epoch_loss = []
        self.accuracy = []

    def forward(self, input):
        x = self.conv1(input.to(self.device))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        logits = self.linear4(x)
        predictions = self.softmax(logits)
        return predictions

    def predict(self, test_loader):
        predicitions = []
        self.eval()
        with torch.no_grad():
            for sample, label in test_loader:
                sample, label = sample.to(self.device), label.to(self.device)
                output = self(sample)
                prediction = output.max(1, keepdim=True)[1]
                predicitions.append(prediction)
        return predicitions

    def train_epoch(self):
        epoch_loss = 0
        correct = 0
        self.train()
        for sample, label in self.train_loader:
            sample, label = sample.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.forward(sample)
            loss = F.nll_loss(output, label)
            epoch_loss += loss.item() * sample.size(0)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(label.view_as(pred)).to(self.device).sum()
            loss.backward()
            self.optimizer.step()
        self.average_epoch_loss.append(epoch_loss / len(self.train_loader.dataset))
        self.accuracy.append(100 * correct / len(self.train_loader.dataset))

    def validate(self, val_loader):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.forward(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).to(self.device).sum()
            test_accuracy = 100 * correct / len(val_loader.dataset)
            return test_loss / len(val_loader.dataset), test_accuracy

    def train_and_validate(self, val_loader):
        test_loss, test_accuracy = [], []
        for i in range(self.epochs):
            self.train_epoch()
            epoch_loss, epoch_accuracy = self.validate(val_loader)
            test_loss.append(epoch_loss)
            test_accuracy.append(epoch_accuracy)
            # self.plot_graphs(test_loss, test_accuracy, i + 1)
            print(f"Train Accuracy is {self.accuracy[-1]} Validation Accuracy is {epoch_accuracy} ")
        self.plot_graphs(test_loss, test_accuracy, self.epochs)

    def train_epochs(self):
        self.train()
        for i in range(self.epochs):
            self.train_epoch()

    def plot_graphs(self, test_loss, test_accuracy, num_epochs):
        """
        Plot loss and accuracy graphs
        :return:
        """
        accuracy = np.array([acc.detach().cpu().numpy() for acc in self.accuracy])
        test_accuracy = np.array([acc.detach().cpu().numpy() for acc in test_accuracy])
        average_epoch_loss = np.array(self.average_epoch_loss)
        test_loss = np.array(test_loss)

        _, plots = plt.subplots(2, 1)
        range_reshaped = np.array(range(1, num_epochs + 1))
        range_reshaped.reshape(1, range_reshaped.shape[0])
        average_epoch_loss.reshape(1, average_epoch_loss.shape[0])
        accuracy.reshape(1, accuracy.shape[0])
        test_accuracy.reshape(1, test_accuracy.shape[0])
        test_loss.reshape(1, test_loss.shape[0])
        plots[0].plot(range_reshaped, average_epoch_loss, label="Train")
        plots[0].plot(range_reshaped, test_loss, label="Validation")
        plots[0].legend()
        plots[0].set_xlabel("Epoch")
        plots[0].set_ylabel("Average Loss")

        plots[1].plot(range_reshaped, accuracy, label="Train")
        plots[1].plot(range_reshaped, test_accuracy, label="Validation")
        plots[1].legend()
        plots[1].set_xlabel("Epoch")
        plots[1].set_ylabel("Accuracy")

        plt.show()


train_dataset = GCommandLoader(sys.argv[1])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)

# val_dataset = GCommandLoader('data/valid')
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True)

test_dataset = GCommandLoader(sys.argv[2])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

test_file_names = [filename[0].split("/")[-1].strip() for filename in test_dataset.spects]
test_file_names = [filename.split("\\")[-1].strip() for filename in test_file_names]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CNNNetwork(train_loader, epochs=30, learning_rate=0.001, device=device)
# model.train_and_validate(val_loader)
model.train_epochs()
predictions = model.predict(test_loader)

predictions_as_classes = [train_dataset.classes[predictions[i]] for i in range(len(predictions))]
test_file_names_num = [int(name.replace('.wav', '')) for name in test_file_names]
zipped_names = list(zip(test_file_names_num, predictions_as_classes))
zipped_names.sort()
predictions_as_classes_sorted = [name for num, name in zipped_names]

zipped_names = list(zip(test_file_names_num, test_file_names))
zipped_names.sort()
test_file_names_sorted = [name for num, name in zipped_names]

# Writing the result to the output file
out_f = open("test_y", "w")
for i in range(len(predictions)):
    out_f.write(f"{test_file_names_sorted[i]},{predictions_as_classes_sorted[i]}\n")
out_f.close()
