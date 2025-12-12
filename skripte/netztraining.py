"""
Skript für das Training eines MLP mit Tabellendaten

@author: Thorsten Clausing
"""
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
import gc
from bewertungsgeber import Bewertungsdatensatz, Bewertungsnetz

# Einlesen der Trainings- und Testdaten
training_liste = []
test_liste = []
zaehler = 0
bewertungen = {}
durchschnitt = 0

with zipfile.ZipFile("../daten/reversi_schwarz.zip", mode="r") as archiv:
    for datei in archiv.namelist():
        if datei.endswith(".of"):
            print(datei)
            with archiv.open(datei, mode="r") as datei_av:
                bewertungen = pickle.load(datei_av)
                for tupel in bewertungen:
                    stellung = np.frombuffer(tupel[0], dtype=np.int8)
                    bewertung = tupel[1][0]/tupel[1][1]
                    eingabe = (torch.from_numpy(stellung)).to(torch.float32)
                    ausgabe = (torch.tensor([bewertung, ])).to(torch.float32)
                    if zaehler % 3:
                        training_liste.append((eingabe, ausgabe))
                    else:
                        test_liste.append((eingabe, ausgabe))
                        durchschnitt += bewertung
                    zaehler += 1

print('Daten geladen: ', len(test_liste) + len(training_liste))
del bewertungen
gc.collect()

# Berechnung des Nenners des Bestimmtheitsmaßes
# für spätere Anpassungstests
durchschnitt /= len(test_liste)
r_nenner = 0
for _, bewertung in test_liste:
    r_nenner += (bewertung.item() - durchschnitt)**2


random.shuffle(training_liste)
training_daten = Bewertungsdatensatz(training_liste)
test_daten = Bewertungsdatensatz(test_liste)
training_datengeber = DataLoader(training_daten, batch_size=32, shuffle=True)
test_datengeber = DataLoader(test_daten, batch_size=32)

modell = Bewertungsnetz()

# Traingsschleife
def train_loop(datengeber, modell, verlustfunktion, optimizer):
    size = len(datengeber.dataset)
    modell.train()      # Unnötig aber 'best practice'
    for batch, (X, y) in enumerate(datengeber):
        # Berechnung von Vorhersage und Verlust
        vorhersage = modell(X)
        verlust = verlustfunktion(vorhersage, y)

        # Backpropagation
        verlust.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Kontrollausdruck des Trainingsverlusts
        if batch % 100000 == 0:
            verlust, current = verlust.item(), batch * 32 + len(X)
            print(f"Verlust: {verlust:>7f}  [{current:>5d}/{size:>5d}]")

# Testschleife
def test_loop(datengeber, modell, verlustfunktion, r_opt):
    modell.eval()       
    num_batches = len(datengeber)
    test_verlust, r_zaehler = 0, 0
    with torch.no_grad():
        for X, y in datengeber:
            vorhersage = modell(X)
            fehler = verlustfunktion(vorhersage, y).item()
            test_verlust += fehler
            r_zaehler += fehler*len(y)           
                        
    test_verlust /= num_batches
    r_quadrat = 1 - r_zaehler/r_nenner
    # Persistente Speicherung der (bisher) besten Gewichte
    if r_quadrat > r_opt:
        r_opt = r_quadrat
        torch.save(modell.state_dict(), 'gewichte_schwarz')
    # Ausdruck des Testergebnisses (Verlust und Bestimmtheitsmaß)
    print(f"Testergebnis\n Durchschnittsverlust: {(test_verlust):>8f}, R_quadrat: {r_quadrat} \n")
    # Rückgabewert: Bisher bester Wert des Bestimmtheitsmaßes
    return r_opt

# Trainingsparameter
optimierer = torch.optim.SGD(modell.parameters(), lr=0.001)
epochen = 5

r_opt = 0.0

# Hauptschleife
for t in range(epochen):
    print(f"Epoche {t+1}\n-------------------------------")
    train_loop(training_datengeber, modell, nn.MSELoss(), optimierer)
    r_opt = test_loop(test_datengeber, modell, nn.MSELoss(), r_opt)

# Ausdruck des besten erreichten Bestimmtheitswerts
print("Bester Bestimmtheitswert: ", r_opt)       


