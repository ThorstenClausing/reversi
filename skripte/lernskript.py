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
from spieler import Lernender_Spieler, Optimierender_Spieler, Stochastischer_Spieler, Minimax_Spieler
from partieumgebung import Partieumgebung

anzahl_partien = 100000
anzahl_tests = 1000
#anzahl_tests_minimax = 10
speicher = Ergebnisspeicher(True, True)

#speicher.bewertung_laden([])
#print('Geladene Bewertungen: ', speicher.anzahl_bewertungen())

spieler_schwarz = Lernender_Spieler(speicher)
spieler_weiss = Lernender_Spieler(speicher)
spieler_opt = Optimierender_Spieler(speicher)
spieler_stoch = Stochastischer_Spieler()
#spieler_minimax = Minimax_Spieler()
partie = Partieumgebung(spieler_schwarz, spieler_weiss, speicher)
test_schwarz_stoch = Partieumgebung(spieler_opt, spieler_stoch)
test_weiss_stoch = Partieumgebung(spieler_stoch, spieler_opt)
#test_schwarz_minimax = Partieumgebung(spieler_opt, spieler_minimax)
#test_weiss_minimax = Partieumgebung(spieler_minimax, spieler_opt)

test_schwarz_stoch.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_schwarz_stoch.test_starten()
test_schwarz_stoch.testprotokoll_drucken()
test_weiss_stoch.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_weiss_stoch.test_starten()
test_weiss_stoch.testprotokoll_drucken()
#test_schwarz_minimax.testprotokoll_zuruecksetzen()
#for _ in range(anzahl_tests_minimax):
#    test_schwarz_minimax.test_starten()
#test_schwarz_minimax.testprotokoll_drucken()
#test_weiss_minimax.testprotokoll_zuruecksetzen()
#for _ in range(anzahl_tests_minimax):
#    test_weiss_minimax.test_starten()
#test_weiss_minimax.testprotokoll_drucken()

for y in [min(i + 2, 10) for i in range(10)]:
    print(y)
    spieler_schwarz.epsilonkehrwert_eingeben(y)
    spieler_weiss.epsilonkehrwert_eingeben(y)
    for _ in range(anzahl_partien):
        partie.partie_starten()
    print('Bewertungen: ', speicher.anzahl_bewertungen())
    test_schwarz_stoch.testprotokoll_zuruecksetzen()
    for _ in range(anzahl_tests):
        test_schwarz_stoch.test_starten()
    test_schwarz_stoch.testprotokoll_drucken()
    test_weiss_stoch.testprotokoll_zuruecksetzen()
    for _ in range(anzahl_tests):
        test_weiss_stoch.test_starten()
    test_weiss_stoch.testprotokoll_drucken()
    #test_schwarz_minimax.testprotokoll_zuruecksetzen()
    #for _ in range(anzahl_tests_minimax):
    #    test_schwarz_minimax.test_starten()
    #test_schwarz_minimax.testprotokoll_drucken()
    #test_weiss_minimax.testprotokoll_zuruecksetzen()
    #for _ in range(anzahl_tests_minimax):
    #    test_weiss_minimax.test_starten()
    #test_weiss_minimax.testprotokoll_drucken()
    
speicher.bewertung_speichern('reversi_8x8_')  