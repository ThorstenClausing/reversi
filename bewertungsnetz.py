from torch import nn
from torch.utils.data import Dataset

class Bewertungsnetz(nn.Module):
    def __init__(self):
        super(Bewertungsnetz, self).__init__()
        self.innere_schicht_eins = nn.Linear(64, 96)
        self.innere_schicht_zwei = nn.Linear(96, 34)
        self.ausgabeschicht = nn.Linear(34, 1)
        self.aktivierung_eins = nn.Tanh()
        self.aktivierung_zwei = nn.Tanh()
        nn.init.xavier_uniform_(self.innere_schicht_eins.weight)
        nn.init.xavier_uniform_(self.innere_schicht_zwei.weight)
        nn.init.xavier_uniform_(self.ausgabeschicht.weight)
        nn.init.zeros_(self.innere_schicht_eins.bias)
        nn.init.zeros_(self.innere_schicht_zwei.bias)
        nn.init.zeros_(self.ausgabeschicht.bias)

    def forward(self, x):
        z = self.innere_schicht_eins(x)
        z = self.aktivierung_eins(z)
        z = self.innere_schicht_zwei(z)
        z = self.aktivierung_zwei(z)
        bewertung = self.ausgabeschicht(z)
        return bewertung

class Bewertungsdaten(Dataset):
    def __init__(self, liste):
        self.liste = liste

    def __len__(self):
        return len(self.liste)

    def __getitem__(self, idx):
        return self.liste[idx][0], self.liste[idx][1]

class Faltendes_Bewertungsnetz(nn.Module):
    def __init__(self):
        super(Faltendes_Bewertungsnetz, self).__init__()
        self.innere_schicht_eins = nn.Conv2d(3, 9, kernel_size=3, padding=1, groups=3)
        self.innere_schicht_zwei = nn.Conv2d(9, 9, kernel_size=3, padding=1, groups=3)
        self.innere_schicht_drei = nn.Linear(576, 300)
        self.ausgabeschicht = nn.Linear(300, 1)
        #self.aktivierung_eins = nn.ReLU()
        #self.aktivierung_zwei = nn.ReLU()
        self.aktivierung_drei = nn.Tanh()
        self.flatten = nn.Flatten()
        nn.init.xavier_uniform_(self.innere_schicht_eins.weight)
        nn.init.xavier_uniform_(self.innere_schicht_zwei.weight)
        nn.init.xavier_uniform_(self.innere_schicht_drei.weight)
        nn.init.xavier_uniform_(self.ausgabeschicht.weight)
        nn.init.zeros_(self.innere_schicht_eins.bias)
        nn.init.zeros_(self.innere_schicht_zwei.bias)
        nn.init.zeros_(self.innere_schicht_drei.bias)
        nn.init.zeros_(self.ausgabeschicht.bias)

    def forward(self, x):
        z = self.innere_schicht_eins(x)
        #z = self.aktivierung_eins(z)
        z = self.innere_schicht_zwei(z)
        #z = self.aktivierung_zwei(z)
        z = self.flatten(z)
        z = self.innere_schicht_drei(z)
        z = self.aktivierung_drei(z)
        bewertung = self.ausgabeschicht(z)
        return bewertung