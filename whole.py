import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import csv

import matplotlib.pyplot as plt

# Device configuration

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print('device:', device)

batch_size = 1000
learning_rate = 0.005
num_epochs = 100000

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = T.Compose( [T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] )

# CIFAR10 dataset
train_dataset = torchvision.datasets.CIFAR10('./cifar10_data', train=True, download=True, transform=transform )
test_dataset = torchvision.datasets.CIFAR10('./cifar10_data', train=False, download=True, transform=transform )

# Data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(train_dataset.data.shape)
print(test_dataset.data.shape)


def convert_to_imshow_format(image):
    # first convert back to [0,1] range from [-1,1] range
    image = image / 2 + 0.5
    image = image.numpy()
    # convert from CHW to HWC
    # from 3x32x32 to 32x32x3
    return image.transpose(1, 2, 0)


dataiter = iter(train_loader)
images, labels = dataiter.next()

images = images[:6]
labels = labels[:6]

fig, axes = plt.subplots(1, len(images), figsize=(12, 2.5))
for idx, image in enumerate(images):
    axes[idx].imshow(convert_to_imshow_format(image))
    axes[idx].set_title(classes[labels[idx]])
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.layer1 = nn.Sequential(
            # N : 미니배치 사이즈
            # Nx3x32x32
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=1),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

        )

        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(10368, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),

            nn.Dropout(p=0.5),
            nn.Linear(300, num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


model = SimpleConvNet()
model.layer1.load_state_dict(torch.load("./layer1"))
model.layer2.load_state_dict(torch.load("./layer2"))
model.fc.load_state_dict(torch.load("./fc"))
model.to(device)

loss_func = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.load_state_dict(torch.load("./optimizer"))

def train(loss_list):
    # train phase
    model.train()

    # create a progress bar
    batch_loss_list = []
    progress = None

    for batch, target in train_loader:
        # Move the training data to the GPU
        batch, target = batch.to(device), target.to(device)

        # forward propagation
        output = model(batch)

        # calculate the loss
        loss = loss_func(output, target)

        # clear previous gradient computation
        optimizer.zero_grad()

        # backpropagate to compute gradients
        loss.backward()

        # update model weights
        optimizer.step()

        # save loss value
        loss_list.append(loss.item())

        # update progress bar
        batch_loss_list.append(loss.item())
        # progress.update(batch.shape[0], sum(batch_loss_list) / len(batch_loss_list))
        progress = sum(batch_loss_list) / len(batch_loss_list)
        print(progress)


def test():
    # test phase
    model.eval()

    correct = 0

    pred_list = []

    # We don't need gradients for test, so wrap in
    # no_grad to save memory
    with torch.no_grad():
        for batch, target in test_loader:
            # Move the training batch to the GPU
            batch, target = batch.to(device), target.to(device)

            # forward propagation
            output = model(batch)

            # get prediction
            output = torch.argmax(output, 1)

            # accumulate correct number
            correct += (output == target).sum().item()

            pred_list.extend(output.tolist())

    # Calculate test accuracy
    acc = 100 * float(correct) / len(test_dataset)
    print('Test accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_dataset), acc))
    torch.save(model.layer1.state_dict(), "./layer1")
    torch.save(model.layer2.state_dict(), "./layer2")
    torch.save(model.fc.state_dict(), "./fc")
    torch.save(optimizer.state_dict(), './optimizer')
    return pred_list


def make_pred_csv():
    pred_list = test()

    print(len(pred_list))

    with open('cifar10_submit.csv', mode='w') as pred_file:
        pred_writer = csv.writer(pred_file, delimiter=',')

        pred_writer.writerow(['id', 'label'])

        for i, label in enumerate(pred_list):
            pred_writer.writerow([i + 1, classes[label]])


loss_list = []

for epoch in range(num_epochs):
    train(loss_list)
    test()


make_pred_csv(model)
