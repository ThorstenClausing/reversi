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

from auswertungsumgebung import Ergebnisspeicher
from spieler import Lernender_Spieler, Optimierender_Spieler, Stochastischer_Spieler
from partieumgebung import Partieumgebung

anzahl_partien = 100000
anzahl_tests = 1000
speicher = Ergebnisspeicher(True, True)

speicher.bewertung_laden()
print('Geladene Bewertungen: ', speicher.anzahl_bewertungen())

spieler_schwarz = Lernender_Spieler(speicher)
spieler_weiss = Lernender_Spieler(speicher)
spieler_opt = Optimierender_Spieler(speicher)
spieler_stoch = Stochastischer_Spieler()
partie = Partieumgebung(spieler_schwarz, spieler_weiss, speicher)
test_schwarz = Partieumgebung(spieler_opt, spieler_stoch)
test_weiss = Partieumgebung(spieler_stoch, spieler_opt)

test_schwarz.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_schwarz.test_starten()
test_schwarz.testprotokoll_drucken()
test_weiss.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_weiss.test_starten()
test_weiss.testprotokoll_drucken()

for y in [7, 8, 9, 10, 10]:
    spieler_schwarz.epsilonkehrwert_eingeben(y)
    spieler_weiss.epsilonkehrwert_eingeben(y)
    for _ in range(anzahl_partien):
        partie.partie_starten()
    print('Bewertungen: ', speicher.anzahl_bewertungen())
    test_schwarz.testprotokoll_zuruecksetzen()
    for _ in range(anzahl_tests):
        test_schwarz.test_starten()
    test_schwarz.testprotokoll_drucken()
    test_weiss.testprotokoll_zuruecksetzen()
    for _ in range(anzahl_tests):
        test_weiss.test_starten()
    test_weiss.testprotokoll_drucken()
    
speicher.bewertung_speichern()  