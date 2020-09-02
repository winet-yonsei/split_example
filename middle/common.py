import pickle
import argparse
import torch.nn as nn


def load_data(data_path):
    with open(data_path, 'rb') as f:
        out = pickle.load(f)
    return out


def save_data(data_path, data):
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)


Parser = argparse.ArgumentParser()
Parser.add_argument(
    '--data',
    type=str,
    help="Input data csv"
)

Parser.add_argument(
    '--device',
    type=str,
    default='cpu',
    help="device name"
)

Parser.add_argument(
    '--train',
    type=bool,
    default=False,
    help="device name"
)

Parser.add_argument(
    '--batch',
    type=int,
    default=1000,
    help="batch size"
)

layer1 = nn.Sequential(
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

layer2 = nn.Sequential(
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


def make_fc_layer(num_classes):
    return nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(10368, 300),
        nn.BatchNorm1d(300),
        nn.ReLU(),

        nn.Dropout(p=0.5),
        nn.Linear(300, num_classes),
        nn.ReLU(),
    )
