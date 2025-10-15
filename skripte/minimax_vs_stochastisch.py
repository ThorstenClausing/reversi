import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

#from auswertungsumgebung import Ergebnisspeicher
from spieler import Alpha_Beta_Spieler
from spieler import Stochastischer_Spieler
from partieumgebung import Partieumgebung

spieler_schwarz = Alpha_Beta_Spieler(4)
spieler_weiss = Stochastischer_Spieler()
partie = Partieumgebung(spieler_schwarz, spieler_weiss, None)
partie.testprotokoll_zuruecksetzen()
for _ in range(1000):
    partie.test_starten()
partie.testprotokoll_drucken()
