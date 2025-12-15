# -*- coding: utf-8 -*-
"""
Skript für Spielstärketests für einen tiefen RL-Reversi-Spieler mit MLP
gegen den Stochastischen Spieler

@author: Thorsten Clausing
"""
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from bewertungsgeber import Bewertungsnetz
from spieler import Optimierender_Spieler, Stochastischer_Spieler
from partieumgebung import Partieumgebung

anzahl_tests = 1000
speicher = Bewertungsnetz()

speicher.load_state_dict(torch.load("Gewichte/gewichte_weiss", weights_only=True))
print('Gewichte geladen.')

spieler_opt = Optimierender_Spieler(speicher)
spieler_stoch = Stochastischer_Spieler()
test_schwarz = Partieumgebung(spieler_opt, spieler_stoch)
test_weiss = Partieumgebung(spieler_stoch, spieler_opt)

test_schwarz.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_schwarz.test_starten()
print("Test schwarz (Weiss):")
test_schwarz.testprotokoll_drucken()
test_weiss.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_weiss.test_starten()
print("Test weiß (Weiss):")
test_weiss.testprotokoll_drucken()
