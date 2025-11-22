import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from spiellogik import Stellung, BRETTGROESSE, als_kanonische_stellung
from tensordict import tensorclass

@tensorclass
class MyData:
    stellungen: torch.Tensor
    bewertungen: torch.Tensor

class Bewertungsnetz(nn.Module):
    
    def __init__(self, schwarz=True, weiss=False, 
                 transformation=als_kanonische_stellung,
                 kanonisch=True,
                 replay_buffer=None, prozessor='cpu', runden=0):
        super(Bewertungsnetz, self).__init__()
        self.innere_schicht_eins = nn.Linear(64, 96)
        self.innere_schicht_zwei = nn.Linear(96, 34)
        self.ausgabeschicht = nn.Linear(34, 1)
        self.aktivierung_eins = nn.Tanh()
        self.aktivierung_zwei = nn.Tanh()
        self.flatten = nn.Flatten()
        nn.init.xavier_uniform_(
            self.innere_schicht_eins.weight)
        nn.init.xavier_uniform_(
            self.innere_schicht_zwei.weight)
        nn.init.xavier_uniform_(
            self.ausgabeschicht.weight)
        nn.init.zeros_(self.innere_schicht_eins.bias)
        nn.init.zeros_(self.innere_schicht_zwei.bias)
        nn.init.zeros_(self.ausgabeschicht.bias)
        self.schwarz = schwarz # Sollen Erfahrungen für Schwarz gespeichert werden?
        self.weiss = weiss     # Sollen Erfahrungen für Weiß gespeichert werden?
        self.transformation = transformation # Wie sollen Stellungen vor Speicherung transformiert werden?
        self.kanonisch = kanonisch # Sollen Stellungen vor Bewertung kanonisiert werden?
        self.replay_buffer = replay_buffer
        self.to(prozessor)
        self.prozessor=prozessor
        # Auf wie viele Nachkommastellen sollen Bewertungen gerundet werden?
        # 0 bedeutet nicht runden.
        self.runden = runden 

    def forward(self, x):
        z = self.flatten(x)
        z = self.innere_schicht_eins(z)
        z = self.aktivierung_eins(z)
        z = self.innere_schicht_zwei(z)
        z = self.aktivierung_zwei(z)
        bewertung = self.ausgabeschicht(z)
        return bewertung
    
    def speichermerkmale_setzen(self, schwarz, weiss):
        self.schwarz = schwarz
        self.weiss = weiss
        
    def rundungsparameter_setzen(self, runden):
        self.runden = runden
    
    def bewertung_geben(self, stellung):
        if self.kanonisch:
            stellung = als_kanonische_stellung(stellung)
            stellung = np.frombuffer(stellung, dtype=np.int8)
        # eingabe = (torch.from_numpy(np.array([stellung]))).to(device, torch.float32)
        with torch.inference_mode():
            eingabe = torch.tensor(
                stellung, dtype=torch.float32, device=self.prozessor).unsqueeze(0)
            ausgabe = self.forward(eingabe).item()
        # Bei untrainiertem Netz sind negative Ausgaben möglich, mit denen die 
        # Spieler nicht umgehen können und die daher abgefangen werden
        # müssen:
        ausgabe = max(0, ausgabe)  
        if self.runden:
            return round(ausgabe, self.runden)
        return ausgabe
    
    def bewertung_aktualisieren(self, protokoll):
      stellung = Stellung()
      stellung.grundstellung()
      zug_nummer = 0
      ergebnis = protokoll.pop()
      liste_stellungen = []
      liste_bewertungen = []
      while protokoll:
          zug_nummer += 1
          zug = protokoll.pop(0)
          stellung.zug_spielen(zug)
          if (zug_nummer % 2 and self.schwarz) or (not zug_nummer % 2 and self.weiss):
              if self.transformation:
                  stellung_neu = self.transformation(stellung)
                  stellung_neu = np.frombuffer(stellung_neu, dtype=np.int8)
              else:
                  stellung_neu = stellung.copy()
              liste_stellungen.append(stellung_neu)
              bewertung = ergebnis[0] if zug_nummer % 2 else ergebnis[1]
              liste_bewertungen.append([bewertung])
      data = MyData(
              stellungen=torch.tensor(
                  np.array(liste_stellungen), 
                  dtype=torch.float32, 
                  device=self.prozessor),
              bewertungen=torch.tensor(
                  np.array(liste_bewertungen), 
                  dtype=torch.float32, 
                  device=self.prozessor), 
              batch_size=[len(liste_stellungen)])
      self.replay_buffer.extend(data)
      
class Grosses_Bewertungsnetz(Bewertungsnetz):
    
    def __init__(self, replay_buffer=None):
        super().__init__(replay_buffer)
        self.innere_schicht_eins = nn.Linear(64, 128)
        self.innere_schicht_zwei = nn.Linear(128, 96)
        self.innere_schicht_drei = nn.Linear(96, 64)
        self.ausgabeschicht = nn.Linear(64, 1)
        self.aktivierung_drei = nn.Tanh()
        nn.init.xavier_uniform_(
            self.innere_schicht_eins.weight)
        nn.init.xavier_uniform_(
            self.innere_schicht_zwei.weight)
        nn.init.xavier_uniform_(
            self.innere_schicht_drei.weight)
        nn.init.xavier_uniform_(
            self.ausgabeschicht.weight)
        nn.init.zeros_(self.innere_schicht_eins.bias)
        nn.init.zeros_(self.innere_schicht_zwei.bias)
        nn.init.zeros_(self.innere_schicht_drei.bias)
        nn.init.zeros_(self.ausgabeschicht.bias)
                
    def forward(self, x):
        z = self.flatten(x)
        z = self.innere_schicht_eins(z)
        z = self.aktivierung_eins(z)
        z = self.innere_schicht_zwei(z)
        z = self.aktivierung_zwei(z)
        z = self.innere_schicht_drei(z)
        z = self.aktivierung_drei(z)
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
    
    def __init__(self, kanonisch=True, runden=0):
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
        self.kanonisch = kanonisch
        # Auf wie viele Nachkommastellen sollen Bewertungen gerundet werden?
        # 0 bedeutet nicht runden.
        self.runden = runden 

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
        if self.kanonisch:
            stellung = als_kanonische_stellung(stellung)
            stellung = np.frombuffer(stellung, dtype=np.int8).reshape(BRETTGROESSE, BRETTGROESSE)
        stellung_plus = np.maximum(stellung, 0)
        stellung_minus = np.maximum(-1*stellung, 0)
        stellung_leer = 1 - stellung_plus - stellung_minus
        stellung_drei_kanaele = np.array([stellung_plus, stellung_minus, stellung_leer])      
        eingabe = (torch.tensor([stellung_drei_kanaele,])).to(torch.float32)
        ausgabe = self.forward(eingabe).item()
        # Bei untrainiertem Netz sind negative Ausgaben möglich, mit denen die 
        # Spieler nicht umgehen können und die daher abgefangen werden
        # müssen:
        ausgabe = max(0, ausgabe)  
        if self.runden:
            return round(ausgabe, self.runden)
        return ausgabe
