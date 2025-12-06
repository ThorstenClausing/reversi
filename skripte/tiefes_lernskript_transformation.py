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
from datetime import datetime
import random
import numpy as np

from spieler import Lernender_Spieler_sigma as Lernender_Spieler
from spieler import Optimierender_Spieler, Stochastischer_Spieler
from bewertungsnetz import Bewertungsnetz
from partieumgebung import Partieumgebung

replay_buffer = td.ReplayBuffer(
    storage=td.LazyTensorStorage(1000000, device='auto'), 
    sampler=td.SamplerWithoutReplacement(shuffle=True),
    batch_size=64)

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
    replay_buffer=replay_buffer)
spieler_schwarz =  Lernender_Spieler(netz)
spieler_weiss = Lernender_Spieler(netz)
umgebung = Partieumgebung(spieler_schwarz, spieler_weiss, netz)

def train_loop(datengeber, modell, verlustfunktion, optimierer, anzahl_schritte):
    modell.train()      # Unnecessary in this situation but added for best practices
    for _ in range(anzahl_schritte):
        stichprobe = datengeber.sample()
        # Compute prediction and loss
        #stichprobe.stellungen.to(prozessor)
        vorhersage = modell(stichprobe.stellungen)
        #stichprobe.bewertungen.to(prozessor)
        verlust = verlustfunktion(vorhersage, stichprobe.bewertungen)
        #print(verlust.item())
        # Backpropagation
        verlust.backward()
        optimierer.step()
        optimierer.zero_grad()

spieler_opt = Optimierender_Spieler(netz)
spieler_stoch = Stochastischer_Spieler()
test_schwarz = Partieumgebung(spieler_opt, spieler_stoch)
test_weiss = Partieumgebung(spieler_stoch, spieler_opt)
        
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

verlustfunktion = nn.MSELoss()
optimierer = torch.optim.SGD(netz.parameters(), lr=0.01)
lernschema = torch.optim.lr_scheduler.ExponentialLR(optimierer, gamma=0.95)

anzahl_partien = 2_500
anzahl_zyklen = 200
anzahl_tests = 1000
anzahl_schritte = 2500
minimum_replay_buffer = 350_000
ergebnisse = []

text = f"""Partien: {anzahl_partien}\nZyklen: {anzahl_zyklen}\nTest: {anzahl_tests} 
Schritte: {anzahl_schritte}\nFüllung Replay-Buffer: {minimum_replay_buffer}"""
print(text)
print("Start", datetime.now().strftime("%H:%M:%S"))

ergebnisse = test_loop(
    test_schwarz, test_weiss, anzahl_tests, ergebnisse)

#print("Start Auffüllen", datetime.now().strftime("%H:%M:%S"))
while len(replay_buffer) < minimum_replay_buffer:
    umgebung.partie_starten()
 
# Äußere Schleife: Abfolge von Trainingszyklen, Beobachtungen generieren,
# einmal Netzparameter anpassen, ggf. einmal Spielstärke testen  
for z in range(anzahl_zyklen):
    #print("Start Spielen", datetime.now().strftime("%H:%M:%S")) 
    # Innere Schleife: neue Beobachtungen generieren und abspeichern
    for _ in range(anzahl_partien):
        umgebung.partie_starten()
    #print("Start Gewichte Trainieren", datetime.now().strftime("%H:%M:%S"))
    train_loop(replay_buffer, netz, verlustfunktion, optimierer, anzahl_schritte)
    lernschema.step()
    if not (z + 1) % 10:
        #print("Start Testen", datetime.now().strftime("%H:%M:%S"))
        ergebnisse = test_loop(
            test_schwarz, test_weiss, anzahl_tests, ergebnisse)
        torch.save(netz.state_dict(), f'tiefe_gewichte_sigsig_trans_6_{z + 1}')

#print("Start Speichern", datetime.now().strftime("%H:%M:%S")) 
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

print("Ende", datetime.now().strftime("%H:%M:%S"))
