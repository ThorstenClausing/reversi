# -*- coding: utf-8 -*-
"""
Skript für einen direkten Spielstärkvergleich zwischen zwei tiefen RL-Reversi-Spielern mit CNN

@author: Thorsten Clausing
"""
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from bewertungsgeber import Faltendes_Bewertungsnetz
from spieler import Optimierender_Spieler
from partieumgebung import Partieumgebung

anzahl_tests = 1000
speicher_eins = Faltendes_Bewertungsnetz()
speicher_zwei = Faltendes_Bewertungsnetz()

speicher_eins.load_state_dict(torch.load("Gewichte/faltende_gewichte_v1", weights_only=True))
speicher_zwei.load_state_dict(torch.load("Gewichte/faltende_gewichte_v2", weights_only=True))
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
