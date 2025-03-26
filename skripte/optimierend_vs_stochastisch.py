import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from auswertungsumgebung import Ergebnisspeicher
from spieler import Lernender_Spieler, Optimierender_Spieler, Stochastischer_Spieler
from partieumgebung import Partieumgebung

anzahl_partien = int(sys.argv[1])
anzahl_tests = int(sys.argv[2])
speicher = Ergebnisspeicher(True, True)
spieler_schwarz = Lernender_Spieler(speicher)
spieler_weiss = Lernender_Spieler(speicher)
spieler_opt = Optimierender_Spieler(speicher)
spieler_stoch = Stochastischer_Spieler()
partie = Partieumgebung(spieler_schwarz, spieler_weiss, speicher)
test_schwarz = Partieumgebung(spieler_opt, spieler_stoch, speicher)
test_weiss = Partieumgebung(spieler_stoch, spieler_opt, speicher)

for y in [2, 3, 4, 5]:
    spieler_schwarz.epsilonkehrwert_eingeben(y)
    spieler_weiss.epsilonkehrwert_eingeben(y)
    for _ in range(anzahl_partien):
        partie.partie_starten()
    test_schwarz.testprotokoll_zuruecksetzen()
    for _ in range(anzahl_tests):
        test_schwarz.test_starten()
    test_schwarz.testprotokoll_drucken()
    test_weiss.testprotokoll_zuruecksetzen()
    for _ in range(anzahl_tests):
        test_weiss.test_starten()
    test_weiss.testprotokoll_drucken()

speicher.bewertung_speichern('neu.of')
