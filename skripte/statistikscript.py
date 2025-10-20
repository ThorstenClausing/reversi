# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 16:46:05 2025

@author: Thorsten
"""
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from spieler import Stochastischer_Spieler
from partieumgebung import Statistikumgebung

spieler_schwarz = Stochastischer_Spieler()
spieler_weiss = Stochastischer_Spieler()
statistik = Statistikumgebung(spieler_schwarz, spieler_weiss)
anzahl_partien = 1000000

statistik.partien_starten(anzahl_partien)