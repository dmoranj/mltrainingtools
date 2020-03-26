import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
from mltrainingtools.cmdlogging import section_logger


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def load_dataset():
    # transforms
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    # datasets
    trainset = torchvision.datasets.FashionMNIST('./data',
                                                 download=True,
                                                 train=True,
                                                 transform=transform)
    testset = torchvision.datasets.FashionMNIST('./data',
                                                download=True,
                                                train=False,
                                                transform=transform)

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=40,
                                              shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=40,
                                             shuffle=False, num_workers=1)

    return trainloader, testloader


def load_network(device):
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net = net.to(device)

    return net, criterion, optimizer


def select_direction(shape):
    direction = np.random.normal(size=shape)
    return direction / np.linalg.norm(direction)


def select_directions(model):
    directions = {}

    for parameter in model.named_parameters():
        direction = select_direction(parameter[1].shape)
        normalized_direction = direction * parameter[1].norm().cpu().detach().numpy()

        directions[parameter[0]] = normalized_direction

    return directions


def perturb(base_parameters, north, south, i, j, step_size):
    new_values = []

    for name, parameter in base_parameters():
        new_value = parameter.cpu().detach().numpy() + i*step_size*north[name] + j*step_size*south[name]
        assert(new_value.shape == parameter.shape)
        new_values.append((name, torch.tensor(new_value)))

    return OrderedDict(new_values)


def estimate_loss(model, device, loss, dataset, new_parameters, max_batch=5):
    model.load_state_dict(new_parameters)

    accumulated_loss = []

    for i, data in enumerate(dataset, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        batch_size = inputs.shape[0]
        predictions = model(inputs)
        loss_value = loss(predictions, labels)
        accumulated_loss.append(loss_value.cpu().detach().numpy()/batch_size)

        if i > max_batch:
            break

    return sum(accumulated_loss)/len(accumulated_loss)


def generate_loss_grid(model, device, loss, dataset, north, south, step_size=0.1, steps=10):
    log = section_logger(1)
    sublog = section_logger(2)

    samples = np.zeros((steps, steps))

    base_parameters = model.named_parameters

    for i in range(steps):
        log('Computing row {}'.format(i))
        for j in range(steps):
            new_parameters = perturb(base_parameters, north, south, i, j, step_size)
            loss_value = estimate_loss(model, device, loss, dataset, new_parameters)
            samples[i, j] = loss_value

    return samples


def plot_landscape(grid):
    pass


def train(model, device, criterion, optimizer, ds_train):
    log = section_logger(1)

    running_loss = 0.0

    for epoch in range(1):  # loop over the dataset multiple times
        log("Training epoch {}".format(epoch))

        for i, data in enumerate(ds_train, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    return model


def execute():
    log = section_logger()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    log("Load dataset")
    ds_train, ds_test = load_dataset()

    log("Load network")
    model, loss, optimizer = load_network(device)

    log("Train network")
    model = train(model, device, loss, optimizer, ds_train)

    log("Sample two random directions")
    north = select_directions(model)
    south = select_directions(model)

    log("Generate grid of loss values")
    grid = generate_loss_grid(model, device, loss, ds_test, north, south)

    log("Plot landscape")
    plot_landscape(grid)


if __name__ == "__main__":
    execute()
