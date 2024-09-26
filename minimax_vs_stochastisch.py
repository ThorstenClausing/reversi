import sys
from auswertungsumgebung import Auswertungsumgebung
from spieler import Minimax_Spieler, Stochastischer_Spieler
from spiellogik import WEISS, SCHWARZ
from partieumgebung import Partieumgebung

anzahl_partien = int(sys.argv[1])
awu = Auswertungsumgebung()
spieler_weiss = Minimax_Spieler(WEISS)
spieler_schwarz = Stochastischer_Spieler(SCHWARZ)
partie = Partieumgebung(spieler_weiss,spieler_schwarz,awu)
for _ in range(anzahl_partien):
  partie.starten()
print(awu.ergebnis_speicher)
