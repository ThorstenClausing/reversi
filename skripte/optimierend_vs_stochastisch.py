import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from auswertungsumgebung import Ergebnisspeicher
from spieler import Optimierender_Spieler
from partieumgebung import Partieumgebung

anzahl_partien = int(sys.argv[1])
speicher = Ergebnisspeicher(True, True)
spieler_schwarz = Optimierender_Spieler(speicher, 5)
spieler_weiss = Optimierender_Spieler(speicher, 5)

partie = Partieumgebung(spieler_schwarz, spieler_weiss, speicher)
for _ in range(anzahl_partien):
  partie.starten()
speicher.bewertung_drucken()
speicher.bewertung_speichern('neu.of')
