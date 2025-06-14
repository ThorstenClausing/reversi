# -*- coding: utf-8 -*-
"""
Created on Sat June 7 2025

@author: Thorsten

"""
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from spieler import Stochastischer_Spieler
from partieumgebung import Partieumgebung_v2 as Partieumgebung

anzahl_tests = 1000
anzahl_runden = 10

spieler_stoch = Stochastischer_Spieler()
test = Partieumgebung(spieler_stoch, spieler_stoch, None)

for _ in range(anzahl_runden):
  test.testprotokoll_zuruecksetzen()
  for _ in range(anzahl_tests):
      test.test_starten()
  test.testprotokoll_drucken()
