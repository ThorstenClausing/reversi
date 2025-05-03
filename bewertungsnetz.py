import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from spiellogik import Stellung, BRETTGROESSE
from tensordict import tensorclass


@tensorclass
class MyData:
    stellungen: torch.Tensor
    bewertungen: torch.Tensor

class Bewertungsnetz(nn.Module):
    def __init__(self, replay_buffer=None):
        super(Bewertungsnetz, self).__init__()
        self.innere_schicht_eins = nn.Linear(64, 96)
        self.innere_schicht_zwei = nn.Linear(96, 34)
        self.ausgabeschicht = nn.Linear(34, 1)
        self.aktivierung_eins = nn.Tanh()
        self.aktivierung_zwei = nn.Tanh()
        self.flatten = nn.Flatten()
        nn.init.xavier_uniform_(self.innere_schicht_eins.weight)
        nn.init.xavier_uniform_(self.innere_schicht_zwei.weight)
        nn.init.xavier_uniform_(self.ausgabeschicht.weight)
        nn.init.zeros_(self.innere_schicht_eins.bias)
        nn.init.zeros_(self.innere_schicht_zwei.bias)
        nn.init.zeros_(self.ausgabeschicht.bias)
        self.replay_buffer = replay_buffer

    def forward(self, x):
        z = self.flatten(x)
        z = self.innere_schicht_eins(z)
        z = self.aktivierung_eins(z)
        z = self.innere_schicht_zwei(z)
        z = self.aktivierung_zwei(z)
        bewertung = self.ausgabeschicht(z)
        return bewertung
    
    def bewertung_geben(self, stellung):
        eingabe = (torch.from_numpy(np.array([stellung]))).to(torch.float32)
        ausgabe = self.forward(self.flatten(eingabe)).item()
        del eingabe
        return ausgabe
    
    def bewertung_aktualisieren(self, protokoll):
      stellung = Stellung()
      stellung.grundstellung()
      zug_nummer = 0
      ergebnis = protokoll.pop()//2
      liste_stellungen = []
      liste_bewertungen = []
      while protokoll:
          zug_nummer += 1
          zug = protokoll.pop(0)
          stellung.zug_spielen(zug)
          liste_stellungen.append(stellung.copy())
          bewertung = ergebnis if zug_nummer % 2 else -1*ergebnis
          liste_bewertungen.append([bewertung])
      data = MyData(
              stellungen=(torch.from_numpy(np.array(liste_stellungen))).to(torch.float32),
              bewertungen=(torch.from_numpy(np.array(liste_bewertungen))).to(torch.float32), 
              batch_size=[len(liste_stellungen)])
      self.replay_buffer.extend(data)

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
        self.innere_schicht_drei = nn.Linear(9*BRETTGROESSE*BRETTGROESSE, 300)
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
    
    def bewertung_geben(self, stellung):
        stellung_plus = np.maximum(stellung, 0)
        stellung_minus = np.maximum(-1*stellung, 0)
        stellung_leer = 1 - stellung_plus - stellung_minus
        stellung_drei_kanaele = np.array([stellung_plus, stellung_minus, stellung_leer])      
        eingabe = (torch.tensor([stellung_drei_kanaele,])).to(torch.float32)
        ausgabe = self.forward(eingabe).item()
        del eingabe
        return ausgabe