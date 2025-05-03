import numpy as np
from spiellogik import Stellung, BRETTGROESSE

ANZAHL_FELDER = BRETTGROESSE**2

class Partieumgebung:

  def __init__(self, spieler_schwarz, spieler_weiss, speicher=None):
    self.spieler_schwarz = spieler_schwarz
    self.spieler_weiss = spieler_weiss    
    self.erfahrungsspeicher = speicher
    self.testprotokoll = None

  def testprotokoll_geben(self):
    return self.testprotokoll

  def testprotokoll_zuruecksetzen(self):
    self.testprotokoll = [0, 0, 0, 0]
    
  def testprotokoll_drucken(self):
    print(self.testprotokoll)
   
  def partie_starten(self):
    stellung = Stellung()
    stellung.grundstellung()
    protokoll = []
    keine_zugmoeglichkeit = False
    zug_nummer = 0
    while True:
        zug_nummer += 1
        if self.__schwarz_am_zug(zug_nummer):
            zug = self.spieler_schwarz.zug_waehlen(stellung)
        else:
            zug = self.spieler_weiss.zug_waehlen(stellung)               
        if zug is None: # Behandlung von Situationen ohne Zugmöglichkeit
            if keine_zugmoeglichkeit:
                zug_nummer -= 1
                break
            keine_zugmoeglichkeit = True
        else:
            keine_zugmoeglichkeit = False 
        stellung.zug_spielen(zug)
        protokoll.append(zug)       
        if zug_nummer >= ANZAHL_FELDER - 4 and np.count_nonzero(stellung) == ANZAHL_FELDER:
            break       
        # Am Ende der Schleife entspricht zug_nummer der Anzahl der tatsächlich
        # gespielten Züge
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
    
  def test_starten(self):
    stellung = Stellung()
    stellung.grundstellung()
    if self.testprotokoll is None:
        self.testprotokoll = [0, 0, 0, 0]
    keine_zugmoeglichkeit = False
    zug_nummer = 0
    while True:
        zug_nummer += 1
        if self.__schwarz_am_zug(zug_nummer):
            zug = self.spieler_schwarz.zug_waehlen(stellung)
        else:
            zug = self.spieler_weiss.zug_waehlen(stellung)
        stellung.zug_spielen(zug)        
        if zug is None: # Behandlung von Situationen ohne Zugmöglichkeit
            if keine_zugmoeglichkeit:
                break
            keine_zugmoeglichkeit = True
        else:
            keine_zugmoeglichkeit = False 
        if zug_nummer >= ANZAHL_FELDER - 4 and np.count_nonzero(stellung) == ANZAHL_FELDER:
            break        
    ergebnis = self.__ergebnis_fuer_schwarz(stellung, zug_nummer)
    self.testprotokoll[0] += ergebnis
    if ergebnis > 0:
      self.testprotokoll[1] += 1
    elif ergebnis == 0:
      self.testprotokoll[2] += 1
    else:
      self.testprotokoll[3] += 1
      
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
          DESCRIPTION. Anzahl der bisher ausgeführten Züge (einschließlich passen)

      Returns 
      -------
      TYPE Integer
      DESCRIPTION. Differenz des nach den WOF-Regeln bestimmten Partieergebnisses

      """
      steindifferenz = stellung.sum()
      # Die Steindifferenze besagt, wie viele Steine der Spieler, der am Zug 
      # ist, mehr hat als sein Gegenspieler.
#      print('Differenz: ', steindifferenz)
      if steindifferenz == 0:
          return 0
      anzahl_leere_felder = ANZAHL_FELDER - np.count_nonzero(stellung)
#     print('Leere Felder: ', anzahl_leere_felder)
      # Leere Felder werden als Punkte für den Spieler gewertet, der mehr
      # Steine hat.
      if steindifferenz > 0:
          ergebnis = steindifferenz + anzahl_leere_felder
      else:
          ergebnis = steindifferenz - anzahl_leere_felder
      # Wenn ungerade viele Züge gespielt wurden, ist Weiß am Zug.
      # Die Steindifferenz besagt dann, wie viele Steine Weiß mehr hat als 
      # Schwarz. Aus Sicht von Schwarz muss das Ergebnis dann umgedreht werden.
      if zug_nummer % 2 == 1: 
          ergebnis = -1*ergebnis
      return ergebnis
