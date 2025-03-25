import numpy as np
from spiellogik import Stellung

class Partieumgebung:

  def __init__(self, spieler_schwarz, spieler_weiss, speicher=None):
    self.spieler_schwarz = spieler_schwarz
    self.spieler_weiss = spieler_weiss    
    self.erfahrungsspeicher = speicher
   
  def partie_starten(self):
    stellung = Stellung()
    stellung.grundstellung()
    protokoll = []
    zu_ende = False
    keine_zugmoeglichkeit = False
    zug_nummer = 1
    while not zu_ende:
      if self.__schwarz_am_zug(zug_nummer):
        zug = self.spieler_schwarz.zug_waehlen(stellung)
      else:
        zug = self.spieler_weiss.zug_waehlen(stellung)
      stellung.zug_spielen(zug)
      protokoll.append(zug)
      zug_nummer += 1
      if zug is None: # Behandlung von Situationen ohne Zugmöglichkeit
        if keine_zugmoeglichkeit:
            protokoll.pop()
            zu_ende = True
        keine_zugmoeglichkeit = True
      else:
        keine_zugmoeglichkeit = False 
      if zug_nummer >= 61 and np.count_nonzero(stellung) == 64:
        zu_ende = True
    ergebnis = self.__ergebnis_fuer_schwarz(stellung, zug_nummer)
    protokoll.append(ergebnis)
    if self.erfahrungsspeicher is not None:
      self.erfahrungsspeicher.bewertung_aktualisieren(protokoll)
    """  
    protokoll.pop()
    e = '\t'
    for zug in protokoll:
      if zug is not None:
         print(zug[0], end=e)
      else:
          print(' pass ', end=e)
      e = '\n' if e == '\t' else '\t'
    print(ergebnis)
    stellung.stellung_anzeigen()
    """
      
  def __schwarz_am_zug(self, zug_nummer):
      return zug_nummer % 2 == 1
  
  def __ergebnis_fuer_schwarz(self, stellung, zug_nummer):
      """
      Rückgabewert: Partieergebnis (Punktdifferenz) aus Sicht von Schwarz

      Parameters
      ----------
      stellung : TYPE Stellung
          DESCRIPTION. Endstellung, für die das Ergebis berechnet werden soll
      zug_nummer : TYPE Integer
          DESCRIPTION. Anzahl der bisher ausgeführten Züge (einschließlich passen)  + 1

      Returns 
      -------
      TYPE Integer
      DESCRIPTION. Differenz des nach den WOF-Regeln bestimmten Partieergebnisses

      """
      steindifferenz = stellung.sum()
#      print('Differenz: ', steindifferenz)
      if steindifferenz == 0:
          return 0
      anzahl_leere_felder = 64 - np.count_nonzero(stellung)
#     print('Leere Felder: ', anzahl_leere_felder)
      if steindifferenz > 0:
          ergebnis = steindifferenz + anzahl_leere_felder
      else:
          ergebnis = steindifferenz - anzahl_leere_felder
      if not zug_nummer % 2:
          ergebnis = -1*ergebnis
      return ergebnis