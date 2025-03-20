import numpy as np
from spiellogik import Stellung
from auswertungsumgebung import Auswertungsumgebung

class Partieumgebung:

  def __init__(self,spieler_weiss,spieler_schwarz,auswertungsumgebung=None):
    self.spieler_weiss = spieler_weiss
    self.spieler_schwarz = spieler_schwarz
    self.auswertungsumgebung = auswertungsumgebung
    self.protokoll = []

  def starten(self):
    stellung = Stellung()
    stellung.grundstellung()
    zu_ende = False
    keine_zugmoeglichkeit = False
    zug_nummer = 1
    while not zu_ende:
      if self.__schwarz_am_zug(zug_nummer):
        zug = self.spieler_schwarz.zug_waehlen(stellung)
      else:
        zug = self.spieler_weiss.zug_waehlen(stellung)
      stellung.zug_spielen(zug)
      self.protokoll.append(zug)
      zug_nummer += 1
      if zug is None: # Behandlung von Situationen ohne Zugmöglichkeit
        if keine_zugmoeglichkeit:
            zu_ende = True
        keine_zugmoeglichkeit = True
      else:
        keine_zugmoeglichkeit = False 
      if zug_nummer >= 61 and np.count_nonzero(stellung) == 64:
        zu_ende = True
    ergebnis = self.ergebnis_fuer_schwarz(stellung, zug_nummer)
    self.protokoll.append(ergebnis)
    if self.auswertungsumgebung != None:
      self.auswertungsumgebung.bewertung_aktualisieren(self.protokoll)
      
  def __scharz_am_zug(zug_nummer):
      return zug_nummer % 2 == 1
  
  def __ergebnis_fuer_schwarz(stellung, zug_nummer):
      steindifferenz = stellung.sum()
      if steindifferenz == 0:
          return 0
      anzahl_leere_felder = 64 - np.count_nonzero(stellung)
      if steindifferenz > 0:
          ergebnis = steindifferenz + anzahl_leere_felder
      else:
          ergebnis = steindifferenz - anzahl_leere_felder
      if not zug_nummer % 2:
          ergebnis = -1*ergebnis
      return ergebnis