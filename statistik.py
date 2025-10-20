# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 17:48:21 2025

@author: Thorsten
"""
import numpy as np
from spiellogik import Stellung, BRETTGROESSE, als_kanonische_stellung, als_schwache_kanonische_stellung

ANZAHL_FELDER = BRETTGROESSE**2

class Statistikumgebung:
        
    def __init__(self,  spieler_schwarz, spieler_weiss):
      self.spieler_schwarz = spieler_schwarz
      self.spieler_weiss = spieler_weiss
      
    def partien_starten(self, anzahl_partien): 
        stellungen = {}
        stellungen_schwach = {}
        fruehestes_passen = 100
        min_laenge = 100
        max_laenge = 0
        sum_laenge = 0
        max_passen = 0
        min_passen = 100
        sum_passen = 0
        sum_kanonisch = 0
        sum_kanonisch_schwach = 0
        for _ in range(anzahl_partien):
            stellung = Stellung()
            stellung.grundstellung()
            zug_nummer = 0
            naechster = "_"
            keine_zugmoeglichkeit = False
            while True:
                zug_nummer += 1
                if self.__schwarz_am_zug(zug_nummer):
                    zug = self.spieler_schwarz.zug_waehlen(stellung)
                    naechster = "w"
                else:
                    zug = self.spieler_weiss.zug_waehlen(stellung)
                    naechster = "s"
                if zug is None: # Behandlung von Situationen ohne Zugmöglichkeit
                    if zug_nummer < fruehestes_passen:
                        fruehestes_passen = zug_nummer
                    if keine_zugmoeglichkeit:
                        zug_nummer -= 1
                        sum_laenge += zug_nummer
                        if zug_nummer < min_laenge:
                            min_laenge = zug_nummer
                        elif zug_nummer > max_laenge:
                            max_laenge = zug_nummer
                        anzahl_passen = zug_nummer - np.count_nonzero(stellung) + 4
                        sum_passen += anzahl_passen
                        if anzahl_passen < min_passen:
                            min_passen = anzahl_passen
                        elif anzahl_passen > max_passen:
                            max_passen = anzahl_passen
                        break
                    keine_zugmoeglichkeit = True
                else:
                    keine_zugmoeglichkeit = False 
                stellung.zug_spielen(zug)
                schluessel = als_kanonische_stellung(stellung)
                schluessel_schwach = als_schwache_kanonische_stellung(stellung)
                if schluessel == stellung.tobytes():
                    sum_kanonisch += 1
                if schluessel_schwach == stellung.tobytes():
                    sum_kanonisch_schwach += 1
                if schluessel not in stellungen.keys():
                    stellungen[schluessel] = [naechster]
                elif naechster not in stellungen[schluessel]:
                    stellungen[schluessel].append(naechster)
                if schluessel_schwach not in stellungen_schwach.keys():
                    stellungen_schwach[schluessel_schwach] = [naechster]
                elif naechster not in stellungen_schwach[schluessel_schwach]:
                    stellungen_schwach[schluessel_schwach].append(naechster)
                if zug_nummer >= ANZAHL_FELDER - 4 and np.count_nonzero(stellung) == ANZAHL_FELDER:
                    sum_laenge += zug_nummer
                    if zug_nummer < min_laenge:
                        min_laenge = zug_nummer
                    elif zug_nummer > max_laenge:
                        max_laenge = zug_nummer
                    anzahl_passen = zug_nummer - np.count_nonzero(stellung) + 4
                    sum_passen += anzahl_passen
                    if anzahl_passen < min_passen:
                        min_passen = anzahl_passen
                    elif anzahl_passen > max_passen:
                        max_passen = anzahl_passen
                    break       
                # Am Ende der Schleife entspricht zug_nummer der Anzahl der tatsächlich
                # gespielten Züge
        print("Anzahl aller Züge:", sum_laenge)
        print("Anzahl aller original kanonischen Stellungen:", sum_kanonisch)
        print("Anzahl unterschiedlicher kanonischer Stellungen:", len(stellungen))
        print("Kürzeste Partie:", min_laenge)
        print("Längste Partie:", max_laenge)
        print("Frühestes Passen:", fruehestes_passen)
        print("Niedrigste Anzahl Passen:", min_passen)
        print("Höchste Anzahl Passen:", max_passen)
        print("Insgesamt gepasst:", sum_passen)
        print("Doppelte kanonische Stellungen:")
        anzahl_doppelte = 0
        for key in stellungen.keys():
            if "w" in stellungen[key] and "s" in stellungen[key]:
                print(key)
                anzahl_doppelte += 1
        print("Anzahl doppelter kanonischer Stellungen:", anzahl_doppelte)
        print("Doppelte schwache kanonische Stellungen:")
        anzahl_doppelte_schwach = 0
        for key in stellungen_schwach.keys():
            if "w" in stellungen_schwach[key] and "s" in stellungen_schwach[key]:
                print(key)
                anzahl_doppelte_schwach += 1
        print("Anzahl doppelter schwacher kanonischer Stellungen:", anzahl_doppelte_schwach)
        print("Ende")
        
    def __schwarz_am_zug(self, zug_nummer):
            return zug_nummer % 2 == 1
        
         
