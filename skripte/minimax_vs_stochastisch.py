import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from auswertungsumgebung import Ergebnisspeicher_v2 as Ergebnisspeicher
from spieler import Minimax_Spieler
from spieler import Stochastischer_Spieler
from partieumgebung import Partieumgebung_v2 as Partieumgebung

spieler_schwarz = Stochastischer_Spieler()
spieler_weiss = Minimax_Spieler()
partie = Partieumgebung(spieler_schwarz, spieler_weiss, None)
partie.testprotokoll_zuruecksetzen()
for _ in range(1000):
    partie.test_starten()
partie.testprotokoll_drucken()
