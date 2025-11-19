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

#prozessor = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
prozessor = 'cpu'
print(f"Prozessor für tiefes Lernskript: {prozessor}")

replay_buffer = td.ReplayBuffer(
    storage=td.LazyTensorStorage(1000000, device='auto'), 
    sampler=td.SamplerWithoutReplacement(shuffle=True),
    batch_size=512)
netz = Bewertungsnetz(
    schwarz=True, 
    weiss=False, 
    replay_buffer=replay_buffer,
    prozessor=prozessor)
spieler_schwarz =  Stochastischer_Spieler() # Lernender_Spieler(netz)
spieler_weiss = Lernender_Spieler(netz)
umgebung = Partieumgebung(spieler_schwarz, spieler_weiss, netz)

#variante = "v2" # Auswahl: v1_, v2, schwarz, weiss
#netz.load_state_dict(torch.load("../Gewichte/gewichte_" + variante, weights_only=True))

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
test_schwarz = None #Partieumgebung(spieler_opt, spieler_stoch)
test_weiss = Partieumgebung(spieler_stoch, spieler_opt)
        
def test_loop(test_schwarz, test_weiss, anzahl_tests, liste, 
              bestes_ergebnis):
    with torch.no_grad():
        #test_schwarz.testprotokoll_zuruecksetzen()
        #for _ in range(anzahl_tests):
        #    test_schwarz.test_starten()
        #ergebnis = test_schwarz.testprotokoll_geben()
        #liste.append(ergebnis)
        #ergebnis_schwarz = (ergebnis[1] + ergebnis[2]/2)/anzahl_tests
        
        test_weiss.testprotokoll_zuruecksetzen()
        for _ in range(anzahl_tests):
            test_weiss.test_starten()
        ergebnis = test_weiss.testprotokoll_geben()
        liste.append(ergebnis)
        ergebnis_weiss = (ergebnis[3] + ergebnis[2]/2)/anzahl_tests
        
        if ergebnis_weiss >= bestes_ergebnis:
            bestes_ergebnis = ergebnis_weiss
            torch.save(netz.state_dict(), 'tiefe_gewichte_weiss')
    return liste, bestes_ergebnis

verlustfunktion = nn.MSELoss()
optimierer = torch.optim.SGD(netz.parameters(), lr=0.005)
lernschema = torch.optim.lr_scheduler.ExponentialLR(optimierer, gamma=0.95)

anzahl_partien = 5_000
anzahl_zyklen = 100
anzahl_tests = 1000
anzahl_schritte = 625
minimum_replay_buffer = 350_000
ergebnisse = []
bestes_ergebnis = 0.8

text = f"""Partien: {anzahl_partien}\nZyklen: {anzahl_zyklen}\nTest: {anzahl_tests} 
Schritte: {anzahl_schritte}\nFüllung Replay-Buffer: {minimum_replay_buffer}"""
print(text)
print("Start", datetime.now().strftime("%H:%M:%S"))

ergebnisse, bestes_ergebnis = test_loop(
    test_schwarz, test_weiss, anzahl_tests, ergebnisse, bestes_ergebnis)

#print("Start Auffüllen", datetime.now().strftime("%H:%M:%S"))
netz.rundungsparameter_setzen(1)
while len(replay_buffer) < minimum_replay_buffer:
    umgebung.partie_starten()
 
# Äußere Schleife: Abfolge von Trainingszyklen, Beobachtungen generieren,
# einmal Netzparameter anpassen, ggf. einmal Spielstärke testen  
for z in range(anzahl_zyklen):
    #print("Start Spielen", datetime.now().strftime("%H:%M:%S")) 
    # Innere Schleife: neue Beobachtungen generieren und abspeichern
    netz.rundungsparameter_setzen(1)
    for _ in range(anzahl_partien):
        umgebung.partie_starten()
    #print("Start Gewichte Trainieren", datetime.now().strftime("%H:%M:%S"))
    train_loop(replay_buffer, netz, verlustfunktion, optimierer, anzahl_schritte)
    lernschema.step()
    if not (z + 1) % 5:
        #print("Start Testen", datetime.now().strftime("%H:%M:%S"))
        netz.rundungsparameter_setzen(0)
        ergebnisse, bestes_ergebnis = test_loop(
            test_schwarz, test_weiss, anzahl_tests, ergebnisse, 
            bestes_ergebnis)

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
