# -*- coding: utf-8 -*-
"""
Skript für den tabularen RL-Reversi-Spieler
Version 3 (Weiß): Stochastischer_Spieler gegen Lernender_Spieler_sigma

@author: Thorsten Clausing
"""
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from bewertungsgeber import Bewertungstabelle
from spieler import Lernender_Spieler_sigma as Lernender_Spieler
from spieler import Optimierender_Spieler, Stochastischer_Spieler
from partieumgebung import Partieumgebung

# Trainingsparameter
anzahl_partien = 100000
anzahl_tests = 1000
anzahl_zyklen = 10

speicher = Bewertungstabelle(False, True)
spieler_schwarz = Stochastischer_Spieler()
spieler_weiss = Lernender_Spieler(speicher)
spieler_opt = Optimierender_Spieler(speicher)
spieler_stoch = Stochastischer_Spieler()
partie = Partieumgebung(spieler_schwarz, spieler_weiss, speicher)
test_weiss = Partieumgebung(spieler_stoch, spieler_opt)

# Spielstärketest mit leerer Tabelle
test_weiss.testprotokoll_zuruecksetzen()
for _ in range(anzahl_tests):
    test_weiss.test_starten()
test_weiss.testprotokoll_drucken()

# Hauptlernzyklus
for _ in range(anzahl_zyklen):
    # Lernpartien
    for _ in range(anzahl_partien):
        partie.partie_starten()
    print('Bewertungen: ', speicher.anzahl_bewertungen())
    # Spielstärketest mit aktualisierter Tabelle
    test_weiss.testprotokoll_zuruecksetzen()
    for _ in range(anzahl_tests):
        test_weiss.test_starten()
    test_weiss.testprotokoll_drucken()
     
# Persistente Speicherung der Bewertungstabelle
speicher.bewertung_speichern('reversi_weiss')  
