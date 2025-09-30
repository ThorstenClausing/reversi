# -*- coding: utf-8 -*-
"""
Created on Tue Sept 30 2025

@author: Thorsten

"""
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from bewertungsnetz import Bewertungsnetz
from spieler import Optimierender_Spieler
from partieumgebung import Partieumgebung

anzahl_tests = 1000
speicher_eins = Bewertungsnetz()
speicher_zwei = Bewertungsnetz()

speicher_eins.load_state_dict(torch.load("Gewichte/gewichte_v1_", weights_only=True))
speicher_zwei.load_state_dict(torch.load("Gewichte/gewichte_v2", weights_only=True))
print('Gewichte geladen.')

spieler_eins = Optimierender_Spieler(speicher_eins)
spieler_zwei = Optimierender_Spieler(speicher_zwei)

test_eins_vs_zwei = Partieumgebung(spieler_eins, spieler_zwei)
test_eins_vs_zwei.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_eins_vs_zwei.test_starten()
print("Test V1 gegen V2:")
test_eins_vs_zwei.testprotokoll_drucken()

test_zwei_vs_eins = Partieumgebung(spieler_zwei, spieler_eins)
test_zwei_vs_eins.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_zwei_vs_eins.test_starten()
print("Test V2 gegen V1:")
test_zwei_vs_eins.testprotokoll_drucken()
