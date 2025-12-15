# -*- coding: utf-8 -*-
"""
Skript für das Training eines tiefen RL-Reversi-Spielers mit intensivierter Trainingsphase

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

from spieler import Lernender_Spieler_sigma as Lernender_Spieler
from spieler import Optimierender_Spieler, Stochastischer_Spieler
from bewertungsgeber import Bewertungsnetz
from partieumgebung import Partieumgebung

replay_buffer = td.ReplayBuffer(
    storage=td.LazyTensorStorage(1000000, device='auto'), 
    sampler=td.SamplerWithoutReplacement(shuffle=True))
netz = Bewertungsnetz(
    schwarz=True, 
    weiss=True, 
    zwischenspeicher=replay_buffer)
spieler_schwarz =  Lernender_Spieler(netz)
spieler_weiss = Lernender_Spieler(netz)
umgebung = Partieumgebung(spieler_schwarz, spieler_weiss, netz)

# Trainingsschleife
def train_loop(datengeber, modell, verlustfunktion, optimierer, anzahl_durchgaenge):
    modell.train()      # Unnötig aber 'best practice'
    # Trainingsspeicher einrichten
    stichprobe = td.ReplayBuffer(
            storage=td.LazyTensorStorage(160256, device='auto'), 
            sampler=td.SamplerWithoutReplacement(shuffle=True),
            batch_size=512)
    # Trainingsspeicher mit Stichprobe aus dem Zwischenspeicher füllen
    stichprobe.extend(datengeber.sample(160256))
    for _ in range(anzahl_durchgaenge):
        for batch in stichprobe:
            optimierer.zero_grad()
            # Vorhersage und Verlust berechnen
            vorhersage = modell(batch.stellungen)
            verlust = verlustfunktion(vorhersage, batch.bewertungen)
            # Backpropagation
            verlust.backward()
            optimierer.step()            

spieler_opt = Optimierender_Spieler(netz)
spieler_stoch = Stochastischer_Spieler()
test_schwarz = Partieumgebung(spieler_opt, spieler_stoch)
test_weiss = Partieumgebung(spieler_stoch, spieler_opt)

# Testschleife
def test_loop(test_schwarz, test_weiss, anzahl_tests, liste):
    with torch.no_grad():
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
anzahl_partien = 5_000
anzahl_zyklen = 100
anzahl_tests = 1_000
anzahl_durchgaenge = 3
minimum_replay_buffer = 350_000
ergebnisse = []
verlustfunktion = nn.MSELoss()
optimierer = torch.optim.SGD(netz.parameters(), lr=0.005)
lernschema = torch.optim.lr_scheduler.ExponentialLR(optimierer, gamma=0.95)

text = f"""Partien: {anzahl_partien}\nZyklen: {anzahl_zyklen}\nTest: {anzahl_tests} 
Durchgänge: {anzahl_durchgaenge}\nFüllung Replay-Buffer: {minimum_replay_buffer}"""
print(text)
print("Start", datetime.now().strftime("%H:%M:%S"))

# Test der Spielstärke vor Trainingsbeginn mit zufaälligen Netzgewichten
ergebnisse = test_loop(
    test_schwarz, test_weiss, anzahl_tests, ergebnisse)

# Auffüllen des Zwischenspeichers
while len(replay_buffer) < minimum_replay_buffer:
    umgebung.partie_starten()
 
# Haupttrainingsschleife  
for z in range(anzahl_zyklen):
    #Erfahrungssammelphase
    for _ in range(anzahl_partien):
        umgebung.partie_starten()
    # Trainingsphase
    train_loop(replay_buffer, netz, verlustfunktion, optimierer, anzahl_durchgaenge)
    lernschema.step()
    if not (z + 1) % 5:
        # Testphase
        ergebnisse = test_loop(
            test_schwarz, test_weiss, anzahl_tests, ergebnisse)
        if z > 55:
            # Persistentes Speichern der aktuellen Netzgewichte
            torch.save(netz.state_dict(), f'tiefe_gewichte_sigsig_5000_mehrfach_{z + 1}')

# Speichern der Trainingsdaten
try:
    with open('tiefes_protokoll.txt', "a") as datei:
        datei.write(text)
        datei.write('\n' + str(datetime.now()) + '\n')
        for ergebnis in ergebnisse:
            datei.write(str(ergebnis) + '\n')
        datei.write('\n')
except:
    for ergebnis in ergebnisse:
        print(*ergebnis)
