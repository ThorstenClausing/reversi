import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import pickle
import zipfile
import gc

liste = []
anzahl_doppelte = 0

with zipfile.ZipFile("../daten/reversi_schwarz.zip", mode="r") as archiv:
    for datei in archiv.namelist():
        with archiv.open(datei, mode="r") as datei_av:
            bewertungen = pickle.load(datei_av)
            for tupel in bewertungen:
                liste.append(tupel[0])

with zipfile.ZipFile("../daten/reversi_weiss.zip", mode="r") as archiv:
    for datei in archiv.namelist():
        with archiv.open(datei, mode="r") as datei_av:
            bewertungen = pickle.load(datei_av)
            for tupel in bewertungen:
                if tupel[0] in liste:
                    anzahl_doppelte += 1
                    print(tupel[0])

print("Anzahl doppelte Stellungen:", anzahl_doppelte)
