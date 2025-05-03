# -*- coding: utf-8 -*-
"""
Created on Sat May  3 13:18:29 2025

@author: User
"""
import torchrl.data

from spieler import Lernender_Spieler
from bewertungsnetz import Bewertungsnetz
from partieumgebung import Partieumgebung

replay_buffer = torchrl.data.ReplayBuffer()
netz = Bewertungsnetz(replay_buffer)
spieler_schwarz = Lernender_Spieler(netz)
spieler_weiss = Lernender_Spieler(netz)
umgebung = Partieumgebung(spieler_schwarz, spieler_weiss, netz)

anzahl_partien = 10

for _ in range(anzahl_partien):
    umgebung.partie_starten()
# Netz mit Daten aus replay_buffer trainieren