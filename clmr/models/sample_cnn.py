import torch
import torch.nn as nn
from .model import Model


class SampleCNN(Model):
    def __init__(self, strides, in_channels=1,  supervised=False, out_dim=10):
        super(SampleCNN, self).__init__()

        self.strides = strides
        self.supervised = supervised
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), stride=(3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.hidden = [
            [128, 128],
            [128, 128],
            [128, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 512],
        ]

        assert len(self.hidden) == len(
            self.strides
        ), "Number of hidden layers and strides are not equal"
        for stride, (h_in, h_out) in zip(self.strides, self.hidden):
            self.sequential.append(
                nn.Sequential(
                    nn.Conv1d(h_in, h_out, kernel_size=stride, stride=1, padding=1),
                    nn.BatchNorm1d(h_out),
                    nn.ReLU(),
                    nn.MaxPool1d(stride, stride=stride),
                )
            )

        # 1 x 512
        self.sequential.append(
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
        )

        self.sequential = nn.Sequential(*self.sequential)

        if self.supervised:
            self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        out = self.sequential(x)
        if self.supervised:
            out = self.dropout(out)

        out = out.reshape(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)
        return logit
