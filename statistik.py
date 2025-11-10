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
      
    def partien_starten(self, anzahl_partien, dateiname='reversi_statistik'): 
        stellungen = {}
        stellungen_kanonisch = {}
        #stellungen_schwach = {}
        fruehestes_passen = 100
        min_laenge = 100
        max_laenge = 0
        sum_laenge = 0
        max_passen = 0
        min_passen = 100
        sum_passen = 0
        sum_kanonisch = 0
        #sum_kanonisch_schwach = 0
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
                schluessel = stellung.tobytes()
                schluessel_kanonisch = als_kanonische_stellung(stellung)
                #schluessel_schwach = als_schwache_kanonische_stellung(stellung)
                if schluessel == schluessel_kanonisch:
                    sum_kanonisch += 1
                #if schluessel == schluessel_schwach:
                #    sum_kanonisch_schwach += 1
                if schluessel not in stellungen.keys():
                    stellungen[schluessel] = [naechster]
                else:
                    stellungen[schluessel].append(naechster)
                if schluessel_kanonisch not in stellungen_kanonisch.keys():
                    stellungen_kanonisch[schluessel_kanonisch] = [naechster]
                else:
                    stellungen_kanonisch[schluessel_kanonisch].append(naechster)
                #if schluessel_schwach not in stellungen_schwach.keys():
                #    stellungen_schwach[schluessel_schwach] = [naechster]
                #elif naechster not in stellungen_schwach[schluessel_schwach]:
                #    stellungen_schwach[schluessel_schwach].append(naechster)
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
        with open(dateiname + '.txt', "w") as datei:
            datei.write("Anzahl aller Züge: " + str(sum_laenge) + '\n')
            datei.write("Anzahl aller original kanonischen Stellungen: " + str(sum_kanonisch) + '\n')
            #datei.write("Anzahl aller original schwach kanonischen Stellungen:", sum_kanonisch_schwach)
            datei.write("Anzahl unterschiedlicher Stellungen: " + str(len(stellungen)) + '\n')
            datei.write("Anzahl unterschiedlicher kanonischer Stellungen: " + str(len(stellungen_kanonisch)) + '\n')
            #datei.write("Anzahl unterschiedlicher schwacher kanonischer Stellungen:", len(stellungen_schwach))
            datei.write("Kürzeste Partie: " + str(min_laenge) + '\n')
            datei.write("Längste Partie: " + str(max_laenge) + '\n')
            datei.write("Frühestes Passen: " + str(fruehestes_passen) + '\n')
            datei.write("Niedrigste Anzahl Passen: " + str(min_passen) + '\n')
            datei.write("Höchste Anzahl Passen: " + str(max_passen) + '\n')
            datei.write("Insgesamt gepasst: " + str(sum_passen) + '\n')
            datei.write("Häufigkeit kanonischer Stellungen:" + '\n')
            anzahl_farbenblinde_kanonisch = 0
            haeufigkeit = [0 for _ in range(anzahl_partien + 1)]
            for key in stellungen_kanonisch.keys():
                wiederholung = len(stellungen_kanonisch[key])
                haeufigkeit[wiederholung] += 1
                if "w" in stellungen_kanonisch[key] and "s" in stellungen_kanonisch[key]:
                    # datei.write(key)
                    anzahl_farbenblinde_kanonisch += 1
                    datei.write(str(stellungen_kanonisch[key]) + '\n\n')
            for i in range(anzahl_partien + 1):
                if haeufigkeit[i] > 0:
                    datei.write(str(i) + " -> " + str(haeufigkeit[i]) + '\n')
            datei.write("Anzahl farbenblinder kanonischer Stellungen: " + str(anzahl_farbenblinde_kanonisch) + '\n')
            doppelte_kanonisch = sum(haeufigkeit[2:anzahl_partien + 1])
            datei.write('Anzahl doppelter kanonischer Stellungen: ' + str(doppelte_kanonisch) + '\n')
        #datei.write("farbenblinde schwache kanonische Stellungen:")
        #anzahl_farbenblinde_schwach = 0
        #for key in stellungen_schwach.keys():
            #if "w" in stellungen_schwach[key] and "s" in stellungen_schwach[key]:
                # datei.write(key)
                #anzahl_farbenblinde_schwach += 1
        #datei.write("Anzahl farbenblinder schwacher kanonischer Stellungen:", anzahl_farbenblinde_schwach)
        #if anzahl_farbenblinde_schwach > 0:
            datei.write("Häufigkeit originaler Stellungen:" + '\n')
            anzahl_farbenblinde = 0
            haeufigkeit = [0 for _ in range(anzahl_partien + 1)]
            # zaehler = 0
            for key in stellungen.keys():
                wiederholung = len(stellungen[key])
                haeufigkeit[wiederholung] += 1
                if "w" in stellungen[key] and "s" in stellungen[key]:
                    #if zaehler < 5:
                    #        datei.write(key)
                    #        zaehler += 1
                    anzahl_farbenblinde += 1 
                    datei.write(str(stellungen[key]) + '\n\n')
            for i in range(anzahl_partien + 1):
                if haeufigkeit[i] > 0:
                    datei.write(str(i) + " -> " + str(haeufigkeit[i]) + '\n')
            datei.write("Anzahl farbenblinder originaler Stellungen: " + str(anzahl_farbenblinde) + '\n')
            doppelte = sum(haeufigkeit[2:anzahl_partien + 1])
            datei.write('Anzahl doppelter originaler Stellungen: ' + str(doppelte))
        print("Ende")
        
    def __schwarz_am_zug(self, zug_nummer):
            return zug_nummer % 2 == 1
        
         
