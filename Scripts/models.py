import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

### Gradient Reversal Layer ###

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(self, x, lambd):
        self.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return grad_output*-self.lambd, None

def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)

### MNIST Architecture ###

class MNIST_Feature_Extractor(nn.Module):
    def __init__(self, input_shape):
        super(MNIST_Feature_Extractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32, 48, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Flatten()
        )

    def forward(self, x):
        return self.conv(x)

class MNIST_Label_Predictor(nn.Module):
    def __init__(self,  n_classes):
        super(MNIST_Label_Predictor, self).__init__()
   
        self.label_predictor = nn.Sequential(
            nn.Linear(1200, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_classes)
        )
    
    def forward(self, x):
        return self.label_predictor(x)

class MNIST_Domain_Classifier(nn.Module):
    def __init__(self):
        super(MNIST_Domain_Classifier, self).__init__()
        self.fc1 = nn.Linear(1200, 100) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)

    def set_lambda(self, lambd):
        self.lambd = lambd
    
    def forward(self, x):
        x = grad_reverse(x, self.lambd)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


### SVHN Architecture ###

class SVHN_Feature_Extractor(nn.Module):
    def __init__(self, input_shape, p = 0.0):
        super(SVHN_Feature_Extractor, self).__init__()
        self.p = p
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(64, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=1),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.Flatten()
        )
    
    def is_training(self, training):
        self.training = training

    def forward(self, x):
        return nn.functional.dropout(self.conv(x), self.p, self.training, inplace=True)

class SVHN_Label_Predictor(nn.Module):
    def __init__(self,  n_classes, p = 0.0):
        super(SVHN_Label_Predictor, self).__init__()
        self.p = p
        self.label_predictor = nn.Sequential(
            nn.Linear(2048, 3072),
            nn.ReLU(),
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.Linear(2048, n_classes)
        )

    def is_training(self, training):
        self.training = training
    
    def forward(self, x):
        return nn.functional.dropout(self.label_predictor(x), self.p, self.training, inplace=True)

class SVHN_Domain_Classifier(nn.Module):
    def __init__(self, p = 0.0):
        super(SVHN_Domain_Classifier, self).__init__()
        self.p = p
        self.fc1 = nn.Linear(2048, 1024) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 1)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def is_training(self, training):
        self.training = training
    
    def forward(self, x):
        x = grad_reverse(x, self.lambd)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return torch.sigmoid(x)