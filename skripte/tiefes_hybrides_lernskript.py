# -*- coding: utf-8 -*-
"""
Skript für das Training eines tiefen RL-Reversi-Spielers mit Netzwerk und Tabelle

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
import numpy as np
from datetime import datetime
import random

from spieler import Lernender_Spieler_sigma as Lernender_Spieler
from spieler import Optimierender_Spieler, Stochastischer_Spieler
from bewertungsgeber import Bewertungstabelle, Bewertungsnetz, BewertungsDaten
from partieumgebung import Partieumgebung

# Netzwerk für die Zugauswahl der Spieler
netz = Bewertungsnetz(
    schwarz=True, 
    weiss=True)
# Tabelle als Zwischenspeicher
tabelle = Bewertungstabelle(
    schwarz=True,
    weiss=True)

spieler_schwarz =  Lernender_Spieler(netz)
spieler_weiss = Lernender_Spieler(netz)
umgebung = Partieumgebung(spieler_schwarz, spieler_weiss, tabelle)

# Trainingsschleife
def train_loop(datengeber, modell, verlustfunktion, optimierer):
    modell.train()      
    for stichprobe in datengeber:
        optimierer.zero_grad()
        # Vorhersage und Verlust berechnen
        vorhersage = modell(stichprobe.stellungen)
        verlust = verlustfunktion(vorhersage, stichprobe.bewertungen)
        # Backpropagation
        verlust.backward()
        optimierer.step()
        
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
anzahl_partien = 2_000
anzahl_zyklen = 250
anzahl_tests = 1000
minimum_daten = 160_000
stichproben_groesse = 160_000
verlustfunktion = nn.MSELoss()
optimierer = torch.optim.SGD(netz.parameters(), lr=0.01)
lernschema = torch.optim.lr_scheduler.ExponentialLR(optimierer, gamma=0.95)

ergebnisse = []
replay_buffer = td.ReplayBuffer(
    storage=td.LazyTensorStorage(stichproben_groesse), 
    sampler=td.SamplerWithoutReplacement(shuffle=True),
    batch_size=64)

text = f"""Partien: {anzahl_partien}\nTest: {anzahl_tests} 
Anzahl Zyklen: {anzahl_zyklen}
Stichprobe {stichproben_groesse}
Füllung Tabelle: {minimum_daten}"""
print(text)
print("Start", datetime.now().strftime("%H:%M:%S"))

# Test der Spielstärke vor Trainingsbeginn mit zufälligen Netzgewichten
ergebnisse = test_loop(
    test_schwarz, test_weiss, anzahl_tests, ergebnisse)

# Auffüllen der Tabelle (= Zwischenspeicher)
while len(tabelle.bewertung) < minimum_daten:
    umgebung.partie_starten()
 
# Haupttrainingsschleife
for z in range(anzahl_zyklen):
    # Erfahrungssammelphase
    for _ in range(anzahl_partien):
        umgebung.partie_starten()
    # Stichprobe aus der Tabelle ziehen und in Replay Buffer geben
    auswahl = random.sample(list(tabelle.bewertung.keys()), stichproben_groesse)
    liste_stellungen = []
    liste_bewertungen = []
    for schluessel in auswahl:
        paar = tabelle.bewertung[schluessel]
        stellung_neu = np.frombuffer(schluessel, dtype=np.int8)
        liste_stellungen.append(stellung_neu)
        bewertung_neu = paar[0]/paar[1]
        liste_bewertungen.append([bewertung_neu])
    daten = BewertungsDaten(
            stellungen=torch.tensor(
                np.array(liste_stellungen), 
                dtype=torch.float32),
            bewertungen=torch.tensor(
                np.array(liste_bewertungen), 
                dtype=torch.float32), 
            batch_size=[len(liste_stellungen)])
    replay_buffer.extend(daten)
    # Trainingsphase
    train_loop(replay_buffer, netz, verlustfunktion, optimierer)
    lernschema.step()
    if not (z + 1) % 10:
        # Testphase
        ergebnisse = test_loop(
            test_schwarz, test_weiss, anzahl_tests, ergebnisse)
        # Persistentes Speichern der aktuellen Netzgewichte
        torch.save(netz.state_dict(), f'tiefe_gewichte_sigsig_2000_hybrid_{z + 1}')

# Speichern der Trainingsdaten 
try:
    with open('tiefes_protokoll.txt', "a") as datei:
        datei.write('Hybrid\n')
        datei.write(text)
        datei.write('\n' + str(datetime.now()) + '\n')
        for ergebnis in ergebnisse:
            datei.write(str(ergebnis) + '\n')
        datei.write('\n')
except:
    for ergebnis in ergebnisse:
        print(*ergebnis)
