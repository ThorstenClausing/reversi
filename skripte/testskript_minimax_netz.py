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
from bewertungsnetz import Bewertungsnetz
from spieler import Optimierender_Spieler, Alpha_Beta_Spieler
from partieumgebung import Partieumgebung

anzahl_tests = 100
speicher = Bewertungsnetz()

gewichte = "Gewichte/tiefe_gewichte_sigsig_hybrid_350"
speicher.load_state_dict(torch.load(gewichte, weights_only=True))
print('Gewichte geladen.', gewichte)

spieler_opt = Optimierender_Spieler(speicher)
tiefe = 6
spieler_minimax = Alpha_Beta_Spieler(tiefe)
print("Alpha-Beta-Tiefe ", tiefe, sep='')
#test_schwarz = Partieumgebung(spieler_opt, spieler_minimax)
test_weiss = Partieumgebung(spieler_minimax, spieler_opt)

#test_schwarz.testprotokoll_zuruecksetzen()
#for _ in range(anzahl_tests):
#    test_schwarz.test_starten()
#print("Test schwarz:")
#test_schwarz.testprotokoll_drucken()
test_weiss.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_weiss.test_starten()
print("Test weiß:")
test_weiss.testprotokoll_drucken()
