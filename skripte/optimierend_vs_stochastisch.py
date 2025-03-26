import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from auswertungsumgebung import Ergebnisspeicher
from spieler import Lernender_Spieler, Optimierender_Spieler, Stochastischer_Spieler
from partieumgebung import Partieumgebung

anzahl_partien = int(sys.argv[1])
speicher = Ergebnisspeicher(True, True)
spieler_schwarz = Lernender_Spieler(speicher, 2)
spieler_weiss = Lernender_Spieler(speicher, 2)

partie = Partieumgebung(spieler_schwarz, spieler_weiss, speicher)
for _ in range(anzahl_partien):
  partie.partie_starten()

speicher.bewertung_speichern('neu.of')

anzahl_tests = int(sys.argv[2])
spieler_opt = Optimierender_Spieler(speicher)
spieler_stoch = Stochastischer_Spieler()

test_schwarz = Partieumgebung(spieler_opt, spieler_stoch, speicher)
for _ in range(anzahl_tests):
  test_schwarz.test_starten()
test_schwarz.testprotokoll_drucken()

test_weiss = Partieumgebung(spieler_stoch, spieler_opt, speicher)
for _ in range(anzahl_tests):
  test_weiss.test_starten()
test_weiss.testprotokoll_drucken()
