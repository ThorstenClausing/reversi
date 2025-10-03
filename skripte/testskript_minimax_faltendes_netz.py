# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 15:09:11 2025

@author: Thorsten

"""
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from bewertungsnetz import Faltendes_Bewertungsnetz
from spieler import Optimierender_Spieler, Minimax_Spieler
from partieumgebung import Partieumgebung

anzahl_tests = 1000
speicher = Faltendes_Bewertungsnetz()

speicher.load_state_dict(torch.load("Gewichte/faltende_gewichte_schwarz"))
print('Gewichte geladen.')

spieler_opt = Optimierender_Spieler(speicher)
spieler_stoch = Minimax_Spieler()
test_schwarz = Partieumgebung(spieler_opt, spieler_stoch)
#test_weiss = Partieumgebung(spieler_stoch, spieler_opt)

test_schwarz.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_schwarz.test_starten()
print("Test schwarz (Gewichte Schwarz[k]):")
test_schwarz.testprotokoll_drucken()
#test_weiss.testprotokoll_zuruecksetzen()
#for _ in range(anzahl_tests):
#    test_weiss.test_starten()
#print("Test weiß (Gewichte Weiss[k]):")
#test_weiss.testprotokoll_drucken()
