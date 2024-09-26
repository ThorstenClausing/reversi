from auswertungsumgebung import Auswertungsumgebung
from spieler import Minimax_Spieler, Stochastischer_Spieler
from spiellogik import WEISS, SCHWARZ
from partieumgebung import Partieumgebung

awu = Auswertungsumgebung()
spieler_weiss = Minimax_Spieler(WEISS)
spieler_schwarz = Stochastischer_Spieler(SCHWARZ)
partie = Partieumgebung(spieler_weiss,spieler_schwarz,awu)
for _ in range(5):
  partie.starten()
print(awu.ergebnis_speicher)
