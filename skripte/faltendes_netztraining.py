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
from bewertungsnetz import Bewertungsdaten, Faltendes_Bewertungsnetz

training_liste = []
test_liste = []
zaehler = 0
bewertungen = {}
durchschnitt = 0

with zipfile.ZipFile("reversi.zip", mode="r") as archiv:
    for datei in archiv.namelist():
        if datei.endswith("20.of"):
            with archiv.open(datei, mode="r") as offene_datei:
                bewertungen = pickle.load(offene_datei)
                for stellung_bytes in bewertungen.keys():
                    stellung = np.frombuffer(stellung_bytes, dtype=np.int8)
                    stellung = stellung.reshape(8,8)
                    stellung_plus = np.maximum(stellung, 0)
                    stellung_minus = np.maximum(-1*stellung, 0)
                    stellung_leer = 1 - stellung_plus - stellung_minus
                    stellung = np.array([stellung_plus, stellung_minus, stellung_leer])
                    bewertung = bewertungen[stellung_bytes][0]/bewertungen[stellung_bytes][1]
                    bewertung_np = np.array([bewertung, ])
                    eingabe = (torch.from_numpy(stellung)).to(torch.float32)
                    ausgabe = (torch.from_numpy(bewertung_np)).to(torch.float32)
                    if zaehler % 3:
                        training_liste.append((eingabe, ausgabe))
                    else:
                        test_liste.append((eingabe, ausgabe))
                        durchschnitt += bewertung
                    zaehler += 1

durchschnitt /= len(test_liste)
print('Daten geladen: ', len(test_liste) + len(training_liste))
r_nenner = 0
for _, bewertung in test_liste:
    r_nenner += (bewertung.item() - durchschnitt)**2

random.shuffle(training_liste)
training_daten = Bewertungsdaten(training_liste)
test_daten = Bewertungsdaten(test_liste)
training_datengeber = DataLoader(training_daten, batch_size=32, shuffle=True)
test_datengeber = DataLoader(test_daten, batch_size=32)
 
modell = Faltendes_Bewertungsnetz()

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


def test_loop(datengeber, modell, verlustfunktion, ergebnisliste):
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
    ergebnisliste.append(r_quadrat)
    print(f"Testergebnis\n Durchschnittsverlust: {(test_loss):>8f}, R_quadrat: {r_quadrat} \n")
    return ergebnisliste

optimierer = torch.optim.Adam(modell.parameters(), lr=0.001)
epochen = 1
ergebnisliste = []
ergebnisliste = test_loop(test_datengeber, modell, nn.MSELoss(), ergebnisliste)
verlustfunktion = nn.MSELoss()
for t in range(epochen):
    print(f"Epoche {t+1}\n-------------------------------")
    train_loop(training_datengeber, modell, verlustfunktion, optimierer)
    ergebnisliste = test_loop(test_datengeber, modell, verlustfunktion, ergebnisliste)

torch.save(modell.state_dict(), 'faltende_gewichte')
for wert in ergebnisliste:
    print(wert)

