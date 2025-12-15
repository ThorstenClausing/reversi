# -*- coding: utf-8 -*-
"""
Skript für das Training eines tiefen RL-Reversi-Spielers mit angereicherten Daten

@author: Thorsten Clausing
"""
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torchrl.data as td
import torch
from torch import nn
from datetime import datetime
import random
import numpy as np

from spieler import Lernender_Spieler_sigma as Lernender_Spieler
from spieler import Optimierender_Spieler, Stochastischer_Spieler
from bewertungsgeber import Bewertungsnetz
from partieumgebung import Partieumgebung

replay_buffer = td.ReplayBuffer(
    storage=td.LazyTensorStorage(1000000, device='auto'), 
    sampler=td.SamplerWithoutReplacement(shuffle=True),
    batch_size=64)

# Funktion zur Anreicherung der Trainingsdaten
def transformieren(stellung):
    stellung_eins = stellung.copy()
    selektor = random.randrange(4)
    match selektor:
        case 1: # 180 Grad nach links rotieren 
            stellung_zwei = np.rot90(stellung_eins, k=2)
            return stellung_zwei.tobytes()
        case 2: # an Nebendiagonale spiegeln
            stellung_zwei = np.rot90(stellung_eins, k=3)
            return np.transpose(stellung_zwei).tobytes()
        case 3: # an Hauptdiagonale spiegeln
            return np.transpose(stellung_eins).tobytes()
        case _:
            return stellung_eins.tobytes()

netz = Bewertungsnetz(
    schwarz=True, 
    weiss=False,
    transformation=transformieren,
    kanonisch=False,
    zwischenspeicher=replay_buffer)
spieler_schwarz =  Lernender_Spieler(netz)
spieler_weiss = Lernender_Spieler(netz)
umgebung = Partieumgebung(spieler_schwarz, spieler_weiss, netz)

# Trainingsschleife
def train_loop(datengeber, modell, verlustfunktion, optimierer, anzahl_schritte):
    modell.train()      # Unnötig aber 'best practice'
    for _ in range(anzahl_schritte):
        stichprobe = datengeber.sample()
        # Vorhersage und Verlust berechnen
        vorhersage = modell(stichprobe.stellungen)
        verlust = verlustfunktion(vorhersage, stichprobe.bewertungen)
        # Backpropagation
        verlust.backward()
        optimierer.step()
        optimierer.zero_grad()

spieler_opt = Optimierender_Spieler(netz)
spieler_stoch = Stochastischer_Spieler()
test_schwarz = Partieumgebung(spieler_opt, spieler_stoch)
test_weiss = Partieumgebung(spieler_stoch, spieler_opt)

# Testschleife
def test_loop(test_schwarz, test_weiss, anzahl_tests, liste):
    with torch.inference_mode():
        test_schwarz.testprotokoll_zuruecksetzen()
        for _ in range(anzahl_tests):
            test_schwarz.test_starten()
        ergebnis = test_schwarz.testprotokoll_geben()
        liste.append(ergebnis)
        test_weiss.testprotokoll_zuruecksetzen()
        for _ in range(anzahl_tests):
            test_weiss.test_starten()
        ergebnis = test_weiss.testprotokoll_geben()
        liste.append(ergebnis)
    return liste

# Trainingsparameter
anzahl_partien = 2_500
anzahl_zyklen = 200
anzahl_tests = 1000
anzahl_schritte = 2500
minimum_replay_buffer = 350_000
ergebnisse = []
verlustfunktion = nn.MSELoss()
optimierer = torch.optim.SGD(netz.parameters(), lr=0.01)
lernschema = torch.optim.lr_scheduler.ExponentialLR(optimierer, gamma=0.95)

text = f"""Partien: {anzahl_partien}\nZyklen: {anzahl_zyklen}\nTest: {anzahl_tests} 
Schritte: {anzahl_schritte}\nFüllung Replay-Buffer: {minimum_replay_buffer}"""
print(text)
print("Start", datetime.now().strftime("%H:%M:%S"))

# Test der Spielstärke vor Trainingsbeginn mit zufälligen Netzgewichten
ergebnisse = test_loop(
    test_schwarz, test_weiss, anzahl_tests, ergebnisse)

# Auffüllen des Zwischenspeichers
while len(replay_buffer) < minimum_replay_buffer:
    umgebung.partie_starten()
 
# Haupttrainingsschleife 
for z in range(anzahl_zyklen):
    # Erfahrungssammelphase
    for _ in range(anzahl_partien):
        umgebung.partie_starten()
    # Trainingsphase
    train_loop(replay_buffer, netz, verlustfunktion, optimierer, anzahl_schritte)
    lernschema.step()
    if not (z + 1) % 10:
        # Testphase
        ergebnisse = test_loop(
            test_schwarz, test_weiss, anzahl_tests, ergebnisse)
        # Persistentes Speichern der aktuellen Netzgewichte
        torch.save(netz.state_dict(), f'tiefe_gewichte_sigsig_trans_6_{z + 1}')

# Speichern der Trainingsdaten 
try:
    with open('tiefes_protokoll.txt', "a") as datei:
        datei.write(text)
        datei.write('\ntransformiert')
        datei.write('\n' + str(datetime.now()) + '\n')
        for ergebnis in ergebnisse:
            datei.write(str(ergebnis) + '\n')
        datei.write('\n')
except:
    for ergebnis in ergebnisse:
        print(*ergebnis)
