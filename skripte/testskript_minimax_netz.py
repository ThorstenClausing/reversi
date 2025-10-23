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

anzahl_tests = 10
speicher = Bewertungsnetz()

variante = "v1_" # Auswahl: v1_, v2, schwarz, weiss
speicher.load_state_dict(torch.load("Gewichte/gewichte_" + variante, weights_only=True))
print('Gewichte geladen.')

spieler_opt = Optimierender_Spieler(speicher)
tiefe = 9
spieler_minimax = Alpha_Beta_Spieler(tiefe)
print("Alpha-Beta-Tiefe ", tiefe, sep='')
#test_schwarz = Partieumgebung(spieler_opt, spieler_minimax)
test_weiss = Partieumgebung(spieler_minimax, spieler_opt)

#test_schwarz.testprotokoll_zuruecksetzen()
#for _ in range(anzahl_tests):
#    test_schwarz.test_starten()
#print("Test schwarz (", variante, "[kanonisch]):", sep='')
#test_schwarz.testprotokoll_drucken()
test_weiss.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_weiss.test_starten()
print("Test weiß (", variante, "[kanonisch]):", sep='')
test_weiss.testprotokoll_drucken()
