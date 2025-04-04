from torch import nn
from torch.utils.data import Dataset

class Bewertungsnetz(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64, 34),
            nn.ReLU(),
            nn.Linear(34, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        bewertung = self.linear_relu_stack(x)
        return bewertung

class Bewertungsdaten(Dataset):
    def __init__(self, liste):
        self.liste = liste

    def __len__(self):
        return len(self.liste)

    def __getitem__(self, idx):
        return self.liste[idx][0], self.liste[idx][1]
