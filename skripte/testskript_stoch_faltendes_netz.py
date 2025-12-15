# -*- coding: utf-8 -*-
"""
Skript für Spielstärketests für einen tiefen RL-Reversi-Spieler mit CNN
gegen den Stochastischen Spieler

@author: Thorsten Clausing
"""
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
from bewertungsgeber import Faltendes_Bewertungsnetz
from spieler import Optimierender_Spieler, Stochastischer_Spieler
from partieumgebung import Partieumgebung

anzahl_tests = 1000

for variante in ["v1_", "v2", "schwarz", "weiss"]:
    speicher = Faltendes_Bewertungsnetz()
    datei = "Gewichte/faltende_gewichte_" + variante
    speicher.load_state_dict(torch.load(datei, weights_only=True))
    print('Faltende Gewichte ', variante, ' geladen.')

    spieler_opt = Optimierender_Spieler(speicher)
    spieler_stoch = Stochastischer_Spieler()
    test_schwarz = Partieumgebung(spieler_opt, spieler_stoch)
    test_weiss = Partieumgebung(spieler_stoch, spieler_opt)

    test_schwarz.testprotokoll_zuruecksetzen()
    for _ in range(anzahl_tests):
        test_schwarz.test_starten()
    print("Test schwarz (", variante, "[k]):")
    test_schwarz.testprotokoll_drucken()
    test_weiss.testprotokoll_zuruecksetzen()
    for _ in range(anzahl_tests):
        test_weiss.test_starten()
    print("Test weiß (", variante, "[k]):")
    test_weiss.testprotokoll_drucken()
