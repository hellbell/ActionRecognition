import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict

class ActionClassifier(nn.Module):
    def __init__(self):
        super(ActionClassifier, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(7*7*512,1024),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(1024,11)
        )

    def forward(self, x):
        x = x.view(-1, 7*7*512)
        x1 = self.fc1(x)
        x2 = self.fc2(x1)
        x3 = self.fc3(x2)
        out = F.log_softmax(x3)

        return out