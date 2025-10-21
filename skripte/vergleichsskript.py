#import sys
#import os

#current = os.path.dirname(os.path.realpath(__file__))
#parent = os.path.dirname(current)
#sys.path.append(parent)

import pickle
import zipfile

print("Start")

liste_eins = []
liste_zwei = []
anzahl_doppelte = 0

with zipfile.ZipFile("../daten/reversi_schwarz.zip", mode="r") as archiv:
    for datei in archiv.namelist():
        with archiv.open(datei, mode="r") as datei_av:
            bewertungen = pickle.load(datei_av)
            for tupel in bewertungen:
                liste_eins.append(tupel[0])
                
print("Schwarz geladen")
liste_eins.sort(reverse=True)
print("Schwarz sortiert")

with zipfile.ZipFile("../daten/reversi_weiss.zip", mode="r") as archiv:
    for datei in archiv.namelist():
        with archiv.open(datei, mode="r") as datei_av:
            bewertungen = pickle.load(datei_av)
            for tupel in bewertungen:
                liste_zwei.append(tupel[0])
                
print("Weiß geladen")
liste_zwei.sort(reverse=True)
print("Weiß sortiert")

eins = liste_eins.pop()
zwei = liste_zwei.pop()
fertig = False

while not fertig:
    while eins < zwei:
        if not liste_eins:
            fertig = True
            break
        else:
            eins = liste_eins.pop()
    while eins > zwei:
        if not liste_zwei:
            fertig = True
            break
        else:
            zwei = liste_zwei.pop()    
    while eins == zwei:
        anzahl_doppelte += 1
        if not (liste_eins and liste_zwei):
            fertig = True
            break
        else:
            eins = liste_eins.pop()
            zwei = liste_zwei.pop()
        

print("Anzahl doppelte Stellungen:", anzahl_doppelte)
print("Ende")
