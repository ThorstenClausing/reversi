"""
Skript für Spielstärketest für einen tabularen RL-Reversi-Spieler
gegen den Stochastischen Spieler

@author: Thorsten Clausing
"""
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from bewertungsgeber import Bewertungstabelle
from spieler import Optimierender_Spieler, Stochastischer_Spieler
from partieumgebung import Partieumgebung

anzahl_tests = 1000
speicher = Bewertungstabelle(True, True)
datei_liste = ['reversi_schwarz_0', 'reversi_schwarz_1']
speicher.bewertung_laden(datei_liste)
print('Geladene Bewertungen: ', speicher.anzahl_bewertungen())

spieler_opt = Optimierender_Spieler(speicher)
spieler_stoch = Stochastischer_Spieler()

test_schwarz = Partieumgebung(spieler_opt, spieler_stoch, speicher)
test_weiss = Partieumgebung(spieler_stoch, spieler_opt, speicher)


test_schwarz.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_schwarz.test_starten()
test_schwarz.testprotokoll_drucken()
test_weiss.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_weiss.test_starten()
test_weiss.testprotokoll_drucken()
