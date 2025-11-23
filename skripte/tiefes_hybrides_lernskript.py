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
import numpy as np
from datetime import datetime, timedelta
import random

from spieler import Lernender_Spieler_sigma as Lernender_Spieler
from spieler import Optimierender_Spieler, Stochastischer_Spieler
from bewertungsnetz import Bewertungsnetz
from partieumgebung import Partieumgebung
from auswertungsumgebung import Ergebnisspeicher
from tensordict import tensorclass

@tensorclass
class BewertungsDaten:
    stellungen: torch.Tensor
    bewertungen: torch.Tensor
    
anfangszeit = datetime.now()
max_dauer = timedelta(hours=24)

netz = Bewertungsnetz(
    schwarz=True, 
    weiss=True)
tabelle = Ergebnisspeicher(
    schwarz=True,
    weiss=True)

spieler_schwarz =  Lernender_Spieler(netz)
spieler_weiss = Lernender_Spieler(netz)
umgebung = Partieumgebung(spieler_schwarz, spieler_weiss, tabelle)

def train_loop(datengeber, modell, verlustfunktion, optimierer):
    modell.train()      
    for stichprobe in datengeber:
        optimierer.zero_grad()
        # Compute prediction and loss
        vorhersage = modell(stichprobe.stellungen)
        verlust = verlustfunktion(vorhersage, stichprobe.bewertungen)
        # Backpropagation
        verlust.backward()
        optimierer.step()
        
spieler_opt = Optimierender_Spieler(netz)
spieler_stoch = Stochastischer_Spieler()
test_schwarz = Partieumgebung(spieler_opt, spieler_stoch)
test_weiss = Partieumgebung(spieler_stoch, spieler_opt)
        
def test_loop(test_schwarz, test_weiss, anzahl_tests, liste, 
              bestes_ergebnis):
    with torch.no_grad():
        test_schwarz.testprotokoll_zuruecksetzen()
        for _ in range(anzahl_tests):
            test_schwarz.test_starten()
        ergebnis = test_schwarz.testprotokoll_geben()
        liste.append(ergebnis)
        ergebnis_schwarz = (ergebnis[1] + ergebnis[2]/2)/anzahl_tests
        if ergebnis_schwarz >= bestes_ergebnis[0]:
            bestes_ergebnis = (ergebnis_schwarz, bestes_ergebnis[1])
            torch.save(netz.state_dict(), 'tiefe_gewichte_sigsig_hybrid_schwarz')
        test_weiss.testprotokoll_zuruecksetzen()
        for _ in range(anzahl_tests):
            test_weiss.test_starten()
        ergebnis = test_weiss.testprotokoll_geben()
        liste.append(ergebnis)
        ergebnis_weiss = (ergebnis[3] + ergebnis[2]/2)/anzahl_tests        
        if ergebnis_weiss >= bestes_ergebnis[1]:
            bestes_ergebnis = (bestes_ergebnis[0], ergebnis_weiss)
            torch.save(netz.state_dict(), 'tiefe_gewichte_sigsig_hybrid_weiss')
    return liste, bestes_ergebnis

verlustfunktion = nn.MSELoss()
optimierer = torch.optim.SGD(netz.parameters(), lr=0.01)
lernschema = torch.optim.lr_scheduler.ExponentialLR(optimierer, gamma=0.95)

anzahl_partien = 2_000
anzahl_zyklen = 0
anzahl_tests = 1000
minimum_daten = 160_000
stichproben_groesse = 160_000
ergebnisse = []
bestes_ergebnis = (0.99, 0.99)
replay_buffer = td.ReplayBuffer(
    storage=td.LazyTensorStorage(stichproben_groesse), 
    sampler=td.SamplerWithoutReplacement(shuffle=True),
    batch_size=64)
zwischenstaende = [150, 200, 230, 250, 275, 300, 325, 350]

text = f"""Partien: {anzahl_partien}\nTest: {anzahl_tests} 
Füllung Tabelle: {minimum_daten}"""
print(text)
print("Start", datetime.now().strftime("%H:%M:%S"))

ergebnisse, bestes_ergebnis = test_loop(
    test_schwarz, test_weiss, anzahl_tests, ergebnisse, bestes_ergebnis)

#print("Start Auffüllen", datetime.now().strftime("%H:%M:%S"))
#netz.rundungsparameter_setzen(1)
while len(tabelle.bewertung) < minimum_daten:
    umgebung.partie_starten()
 
# Äußere Schleife: Abfolge von Trainingszyklen, Beobachtungen generieren,
# einmal Netzparameter anpassen, ggf. einmal Spielstärke testen  
while True:
    anzahl_zyklen += 1
    #print("Start Spielen", datetime.now().strftime("%H:%M:%S")) 
    # Innere Schleife: neue Beobachtungen generieren und abspeichern
    #netz.rundungsparameter_setzen(1)
    for _ in range(anzahl_partien):
        umgebung.partie_starten()
    #print("Start Gewichte Trainieren", datetime.now().strftime("%H:%M:%S"))
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
    train_loop(replay_buffer, netz, verlustfunktion, optimierer)
    lernschema.step()
    if not anzahl_zyklen % 10:
        #print("Start Testen", datetime.now().strftime("%H:%M:%S"))
        #netz.rundungsparameter_setzen(0)
        ergebnisse, bestes_ergebnis = test_loop(
            test_schwarz, test_weiss, anzahl_tests, ergebnisse, 
            bestes_ergebnis)
    if anzahl_zyklen in zwischenstaende:
        torch.save(netz.state_dict(), f'tiefe_gewichte_sigsig_hybrid_{anzahl_zyklen}')
        try:
            with open('tiefes_protokoll.txt', "a") as datei:
                datei.write(f'Zwischenstand nach {anzahl_zyklen} Zyklen:\n')
                datei.write('Hybrid\n')
                datei.write(text)
                datei.write('\n' + str(datetime.now()) + '\n')
                for ergebnis in ergebnisse:
                    datei.write(str(ergebnis) + '\n')
                datei.write('\n')
        except:
            for ergebnis in ergebnisse:
                print(*ergebnis)
    zeit = datetime.now()
    if zeit - anfangszeit > max_dauer:
        break

torch.save(netz.state_dict(), 'tiefe_gewichte_sigsig_hybrid_final')

#print("Start Speichern", datetime.now().strftime("%H:%M:%S")) 
try:
    with open('tiefes_protokoll.txt', "a") as datei:
        datei.write(f'Hybrid nach {anzahl_zyklen} Zyklen\n')
        datei.write(text)
        datei.write('\n' + str(datetime.now()) + '\n')
        for ergebnis in ergebnisse:
            datei.write(str(ergebnis) + '\n')
        datei.write('\n')
except:
    for ergebnis in ergebnisse:
        print(*ergebnis)

print("Ende", datetime.now().strftime("%H:%M:%S"))
