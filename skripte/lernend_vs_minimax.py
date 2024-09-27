import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from auswertungsumgebung import Auswertungsumgebung
from spieler import Lernender_Spieler, Minimax_Spieler
from spiellogik import WEISS, SCHWARZ
from partieumgebung import Partieumgebung

anzahl_partien = int(sys.argv[1])
awu = Auswertungsumgebung()
spieler_weiss = Lernender_Spieler(WEISS,awu)
spieler_schwarz = Minimax_Spieler(SCHWARZ)
partie = Partieumgebung(spieler_weiss,spieler_schwarz,awu)
for _ in range(anzahl_partien):
  partie.starten()
print(awu.ergebnis_speicher)
awu.bewertung_speichern()
