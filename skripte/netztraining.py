import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from torch.utils.data import DataLoader
from torch import nn
import pickle
import random
import numpy as np
import zipfile
from bewertungsnetz import Bewertungsdaten, Bewertungsnetz

training_liste = []
test_liste = []
zaehler = 0
bewertungen = {}
durchschnitt = 0

with zipfile.ZipFile("reversi.zip", mode="r") as archiv:
    for datei in archiv.namelist():
        if datei.endswith("20.of"):
            with archiv.open(datei, mode="r") as datei_av:
                bewertungen = pickle.load(datei_av)
                for stellung_bytes in bewertungen.keys():
                    stellung = np.frombuffer(stellung_bytes, dtype=np.int8)
                    bewertung = bewertungen[stellung_bytes][0]/bewertungen[stellung_bytes][1]
                    eingabe = (torch.from_numpy(stellung)).to(torch.float32)
                    ausgabe = (torch.tensor([bewertung, ])).to(torch.float32)
                    if zaehler % 3:
                        training_liste.append((eingabe, ausgabe))
                    else:
                        test_liste.append((eingabe, ausgabe))
                        durchschnitt += bewertung
                    zaehler += 1

durchschnitt /= len(test_liste)

r_nenner = 0
for _, bewertung in test_liste:
    r_nenner += (bewertung.item() - durchschnitt)**2

random.shuffle(training_liste)
training_daten = Bewertungsdaten(training_liste)
test_daten = Bewertungsdaten(test_liste)
training_datengeber = DataLoader(training_daten, batch_size=32, shuffle=True)
test_datengeber = DataLoader(test_daten, batch_size=32)

modell = Bewertungsnetz()

def train_loop(datengeber, modell, verlustfunktion, optimizer):
    size = len(datengeber.dataset)
    modell.train()      # Unnecessary in this situation but added for best practices
    for batch, (X, y) in enumerate(datengeber):
        # Compute prediction and loss
        vorhersage = modell(X)
        verlust = verlustfunktion(vorhersage, y)

        # Backpropagation
        verlust.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            verlust, current = verlust.item(), batch * 32 + len(X)
            print(f"Verlust: {verlust:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(datengeber, modell, verlustfunktion):
    modell.eval()       # Unnecessary in this situation but added for best practices
    num_batches = len(datengeber)
    test_loss, r_zaehler = 0, 0
    with torch.no_grad():
        for X, y in datengeber:
            vorhersage = modell(X)
            fehler = verlustfunktion(vorhersage, y).item()
            test_loss += fehler
            r_zaehler += fehler*len(y)
            
                        
    test_loss /= num_batches
    r_quadrat = 1 - r_zaehler/r_nenner
    print(f"Testergebnis\n Durchschnittsverlust: {(test_loss):>8f}, R_quadrat: {r_quadrat} \n")

optimierer = torch.optim.Adam(modell.parameters(), lr=0.001)
epochen = 20
for t in range(epochen):
    print(f"Epoche {t+1}\n-------------------------------")
    train_loop(training_datengeber, modell, nn.MSELoss(), optimierer)
    test_loop(test_datengeber, modell, nn.MSELoss())

torch.save(modell.state_dict(), 'gewichte')

print("Ende")

