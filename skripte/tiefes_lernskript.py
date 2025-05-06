# -*- coding: utf-8 -*-
"""
Created on Sat May  3 13:18:29 2025

@author: Thorsten
"""
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torchrl.data as td
import torch
from torch import nn
from spieler import Lernender_Spieler, Optimierender_Spieler, Stochastischer_Spieler
from bewertungsnetz import Bewertungsnetz
from partieumgebung import Partieumgebung

replay_buffer = td.ReplayBuffer(storage=td.LazyMemmapStorage(1000000), batch_size=128)
netz = Bewertungsnetz(replay_buffer)
spieler_schwarz = Lernender_Spieler(netz)
spieler_weiss = Lernender_Spieler(netz)
umgebung = Partieumgebung(spieler_schwarz, spieler_weiss, netz)

def train_loop(datengeber, modell, verlustfunktion, optimizer, anzahl_schritte):
    modell.train()      # Unnecessary in this situation but added for best practices
    for batch, stichprobe in enumerate(datengeber):
        # Compute prediction and loss
        vorhersage = modell(stichprobe.stellungen)
        verlust = verlustfunktion(vorhersage, stichprobe.bewertungen)
        #print(verlust.item())
        # Backpropagation
        verlust.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch >= anzahl_schritte - 1: 
            break

spieler_opt = Optimierender_Spieler(netz)
spieler_stoch = Stochastischer_Spieler()
test_schwarz = Partieumgebung(spieler_opt, spieler_stoch)
test_weiss = Partieumgebung(spieler_stoch, spieler_opt)
        
def test_loop(modell, test_schwarz, test_weiss, anzahl_tests, liste):
    modell.eval()
    test_schwarz.testprotokoll_zuruecksetzen()
    for _ in range(anzahl_tests):
        test_schwarz.test_starten()
    liste.append(test_schwarz.testprotokoll_geben())
    test_weiss.testprotokoll_zuruecksetzen()
    for _ in range(anzahl_tests):
        test_weiss.test_starten()
    liste.append(test_weiss.testprotokoll_geben())
    return liste

verlustfunktion = nn.MSELoss()
optimator = torch.optim.Adam(netz.parameters(), lr=0.001)

anzahl_partien = 10
anzahl_zyklen = 10
ergebnisse = []

ergebnisse = test_loop(netz, test_schwarz, test_weiss, 1000, ergebnisse)
# Äußere Schleife: Abfolge von Trainingszyklen, einmal Netzparameter anpassen,
# einmal Spielstärke testen
for epsilon_kw in [min(2 + i, 10) for i in range(anzahl_zyklen)]:
    spieler_schwarz.epsilonkehrwert_eingeben(epsilon_kw)
    spieler_weiss.epsilonkehrwert_eingeben(epsilon_kw)
    # Innere Schleife: neue Beobachtungen generieren und abspeichern
    for _ in range(anzahl_partien):
        umgebung.partie_starten()
    train_loop(replay_buffer, netz, verlustfunktion, optimator, 1000)
    ergebnisse = test_loop(netz, test_schwarz, test_weiss, 1000, ergebnisse)
for ergebnis in ergebnisse:
    print(*ergebnis)
