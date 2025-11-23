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

from spieler import Lernender_Spieler_sigma as Lernender_Spieler
from spieler import Optimierender_Spieler, Stochastischer_Spieler
from bewertungsnetz import Bewertungsnetz
from partieumgebung import Partieumgebung

replay_buffer = td.ReplayBuffer(
    storage=td.LazyTensorStorage(1000000, device='auto'), 
    sampler=td.SamplerWithoutReplacement(shuffle=True))
netz = Bewertungsnetz(
    schwarz=True, 
    weiss=True, 
    replay_buffer=replay_buffer)
spieler_schwarz =  Lernender_Spieler(netz)
spieler_weiss = Lernender_Spieler(netz)
umgebung = Partieumgebung(spieler_schwarz, spieler_weiss, netz)

def train_loop(datengeber, modell, verlustfunktion, optimierer, anzahl_durchgaenge):
    modell.train()      # Unnecessary in this situation but added for best practices
    stichprobe = td.ReplayBuffer(
            storage=td.LazyTensorStorage(160000, device='auto'), 
            sampler=td.SamplerWithoutReplacement(shuffle=True),
            batch_size=64)
    stichprobe.extend(datengeber.sample(160000))
    for _ in range(anzahl_durchgaenge):
        for batch in stichprobe:
            optimierer.zero_grad()
            # Compute prediction and loss
            vorhersage = modell(batch.stellungen)
            verlust = verlustfunktion(vorhersage, batch.bewertungen)
            # Backpropagation
            verlust.backward()
            optimierer.step()            

spieler_opt = Optimierender_Spieler(netz)
spieler_stoch = Stochastischer_Spieler()
test_schwarz = Partieumgebung(spieler_opt, spieler_stoch)
test_weiss = Partieumgebung(spieler_stoch, spieler_opt)
        
def test_loop(test_schwarz, test_weiss, anzahl_tests, liste, bestes_ergebnis):
    with torch.no_grad():
        test_schwarz.testprotokoll_zuruecksetzen()
        for _ in range(anzahl_tests):
            test_schwarz.test_starten()
        ergebnis = test_schwarz.testprotokoll_geben()
        liste.append(ergebnis)
        ergebnis_schwarz = (ergebnis[1] + ergebnis[2]/2)/anzahl_tests
        if ergebnis_schwarz >= bestes_ergebnis[0]:
            bestes_ergebnis = (ergebnis_schwarz, bestes_ergebnis[1])
            torch.save(netz.state_dict(), 'tiefe_gewichte_sigsig_mehrfach_schwarz')
        test_weiss.testprotokoll_zuruecksetzen()
        for _ in range(anzahl_tests):
            test_weiss.test_starten()
        ergebnis = test_weiss.testprotokoll_geben()
        liste.append(ergebnis)
        ergebnis_weiss = (ergebnis[3] + ergebnis[2]/2)/anzahl_tests
        if ergebnis_weiss >= bestes_ergebnis[1]:
            bestes_ergebnis = (bestes_ergebnis[0], ergebnis_weiss)
            torch.save(netz.state_dict(), 'tiefe_gewichte_sigsig_mehrfach_weiss')
    return liste, bestes_ergebnis

verlustfunktion = nn.MSELoss()
optimierer = torch.optim.SGD(netz.parameters(), lr=0.01)
lernschema = torch.optim.lr_scheduler.ExponentialLR(optimierer, gamma=0.95)

anzahl_partien = 2_000
anzahl_zyklen = 500
anzahl_tests = 1_000
anzahl_durchgaenge = 3
minimum_replay_buffer = 350_000
ergebnisse = []
bestes_ergebnis = (0.99, 0.99)

text = f"""Partien: {anzahl_partien}\nZyklen: {anzahl_zyklen}\nTest: {anzahl_tests} 
Durchgänge: {anzahl_durchgaenge}\nFüllung Replay-Buffer: {minimum_replay_buffer}"""
print(text)
print("Start", datetime.now().strftime("%H:%M:%S"))

ergebnisse, bestes_ergebnis = test_loop(
    test_schwarz, test_weiss, anzahl_tests, ergebnisse, bestes_ergebnis)

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
    train_loop(replay_buffer, netz, verlustfunktion, optimierer, anzahl_durchgaenge)
    lernschema.step()
    if not (z + 1) % 10:
        #print("Start Testen", datetime.now().strftime("%H:%M:%S"))
        ergebnisse, bestes_ergebnis = test_loop(
            test_schwarz, test_weiss, anzahl_tests, ergebnisse, bestes_ergebnis)
        if z + 1 == 250:
            torch.save(netz.state_dict(), 'tiefe_gewichte_sigsig_mehrfach_500')

torch.save(netz.state_dict(), 'tiefe_gewichte_sigsig_mehrfach_final')
replay_buffer.dumps('replay_buffer')

#print("Start Speichern", datetime.now().strftime("%H:%M:%S")) 
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

print("Ende", datetime.now().strftime("%H:%M:%S"))
