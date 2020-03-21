import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

class DataUtils:
    def __init__(self):
        self.X, self.y = load_iris(return_X_y=True)

    def split_data(self, test_size=0.2, random_state=42, shuffle=True):
        train_X, test_X, train_y, test_y = train_test_split(self.X, self.y, random_state=random_state, shuffle=shuffle, test_size=test_size)
        train_X, test_X = Variable(torch.from_numpy(train_X)).float(), Variable(torch.from_numpy(test_X)).float()
        train_y, test_y = Variable(torch.from_numpy(train_y)).long(), Variable(torch.from_numpy(test_y)).long()

        return train_X, test_X, train_y, test_y

    def plot(self, data, label, title, fig_name):
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(label)
        plt.plot(list(range(len(data))), data)
        plt.savefig(fig_name)
        plt.clf()

class Trainer:
    def __init__(self, model, n_epochs, learning_rate, data_utils):
        self.model = model
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.data_utils = data_utils
        self.loss_arr = []
        self.accuracy_arr = []

    def train(self, plot_results=False, print_metrics=False):
        self.train_X, self.test_X, self.train_y, self.test_y = self.data_utils.split_data()

        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            out = self.model(self.train_X)
            loss_ = loss(out, self.train_y)
            loss_.backward()
            optimizer.step()

            predictions = self.model(self.test_X)
            _, predictions = torch.max(predictions, dim=1)

            acc = accuracy_score(self.test_y, predictions)

            self.loss_arr.append(loss_)
            self.accuracy_arr.append(acc)

            if epoch % 100 == 0:
                print('Epoch: {}, loss: {}'.format(epoch, loss_))

        if plot_results:
            self.plot_loss_acc()

        if print_metrics:
            self.print_model_metrics()

    def plot_loss_acc(self):
        print('** SAVING PLOTS **')
        self.data_utils.plot(self.loss_arr, label='Accuracy', title='Accuracy plot', fig_name='accuracy_plot.png')
        self.data_utils.plot(self.accuracy_arr, label='Loss', title='Loss plot', fig_name='loss_plot.png')

    def print_model_metrics(self):
        print('Accuracy: {}'.format(self.accuracy_arr[-1]))

        predictions = self.model(self.test_X)
        _, predictions = torch.max(predictions, dim=1)

        print('Macro recall: {}, macro precision: {}'.format(recall_score(self.test_y, predictions.data, average='macro'), precision_score(self.test_y, predictions.data, average='macro')))
        print('Micro recall: {}, micro precision: {}'.format(recall_score(self.test_y, predictions.data, average='micro'), precision_score(self.test_y, predictions.data, average='micro')))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.init_layers()
        self.init_activations()

    def init_activations(self):
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def init_layers(self):
        self.linear1 = nn.Linear(4, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 64)
        self.linear4 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.softmax(self.linear4(x))
        return x

if __name__ == '__main__':
    data_utils = DataUtils()
    model = Model()
    trainer = Trainer(model=model, n_epochs=501, learning_rate=.01, data_utils=data_utils)
    trainer.train(plot_results=True, print_metrics=True)
