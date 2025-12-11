import numpy as np
from spiellogik import Stellung, BRETTGROESSE

ANZAHL_FELDER = BRETTGROESSE**2

 
class Partieumgebung:
    """
    Simuliert und verwaltet Reversi-Partien.

    Diese Klasse koordiniert den Ablauf einer Partie, lässt Spieler Züge wählen,
    aktualisiert den Spielzustand und ermittelt das Spielergebnis. Sie kann auch
    Testpartien durchführen und deren Ergebnisse protokollieren.

    Attribute
    ----------
    spieler_schwarz : object
        Das Spielerobjekt, das die Züge für Schwarz wählt.
    spieler_weiss : object
        Das Spielerobjekt, das die Züge für Weiß wählt.
    speicher : object, optional
        Ein Objekt, das zum Speichern von Spieldaten verwendet wird. 
        Muss eine Methode `bewertung_aktualisieren(protokoll)` 
        implementieren. Der Standardwert ist None.
    testprotokoll : list of (tuple, int, int, int) or None
        Protokoll für Testpartien. Wird bei der ersten Testpartie initialisiert.
        Format: `[(summe_punkte_schwarz, summe_punkte_weiss), siege_schwarz, unentschieden, siege_weiss]`.
        Initial ist es `None`.
    """    
    def __init__(self,  spieler_schwarz, spieler_weiss, speicher=None):
      """
      Initialisiert eine neue Partieumgebung mit den gegebenen Spielern und optionalem Speicher.

      Parameters
      ----------
      spieler_schwarz : object
          Der Spieler, der mit den schwarzen Steinen spielt.
      spieler_weiss : object
          Der Spieler, der mit den weißen Steinen spielt.
      speicher : object, optional
          Ein Objekt zum Speichern von Bewertungen basierend auf gespielten Partien. 
          Der Standardwert ist None.
      """
      self.spieler_schwarz = spieler_schwarz
      self.spieler_weiss = spieler_weiss
      self.speicher = speicher
    
    def partie_starten(self):
      """
      Startet eine komplette Partie und führt sie bis zum Ende durch.

      Die Methode initialisiert eine neue Stellung, lässt die Spieler abwechselnd
      Züge wählen und spielen, bis keine Züge mehr möglich sind. Das Protokoll der Partie
      (gespielte Züge und Endergebnis) wird, falls vorhanden, an das `speicher`-Objekt
      übergeben.
      """
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
      ergebnis = self.__ergebnis(stellung, zug_nummer)
      protokoll.append(ergebnis)
      if self.speicher is not None:
          self.speicher.bewertung_aktualisieren(protokoll)
          
    def test_starten(self):
      """
      Startet eine Testpartie und aktualisiert das interne Testprotokoll.

      Diese Methode funktioniert ähnlich wie `partie_starten`, jedoch wird kein
      Partieprotokoll an einen externen Speicher übergeben, sondern das Partieergebnis
      statistisch in `self.testprotokoll` erfasst.
      """
      stellung = Stellung()
      stellung.grundstellung()
      if self.testprotokoll is None:
          self.testprotokoll = [(0, 0), 0, 0, 0]
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
      ergebnis = self.__ergebnis(stellung, zug_nummer)
      self.testprotokoll[0] = (
          self.testprotokoll[0][0] + ergebnis[0], self.testprotokoll[0][1] + ergebnis[1])
      if ergebnis[0] > ergebnis[1]:
        self.testprotokoll[1] += 1
      elif ergebnis[0] == ergebnis[1]:
        self.testprotokoll[2] += 1
      else:
        self.testprotokoll[3] += 1

    def testprotokoll_zuruecksetzen(self):
      """
      Setzt das interne Testprotokoll auf Null zurück.  
      """
      self.testprotokoll = [(0, 0), 0, 0, 0]
              
    def testprotokoll_geben(self):
      """
      Gibt das aktuelle Testprotokoll zurück.

      Rückgabe
      -------
      list of (tuple, int, int, int) or None
          Das Testprotokoll im Format:
          `[(summe_punkte_schwarz, summe_punkte_weiss), siege_schwarz, unentschieden, siege_weiss]`.
          Gibt `None` zurück, wenn noch keine Testpartie gestartet wurde.
      """
      return self.testprotokoll
     
    def testprotokoll_drucken(self):
      """
      Druckt das aktuelle Testprotokoll auf die Konsole.
      """
      print(self.testprotokoll)
        
    def __schwarz_am_zug(self, zug_nummer):
      """
      Überprüft, ob Schwarz (der erste Spieler) am Zug ist.

      Parameters
      ----------
      zug_nummer : int
          Die aktuelle Zugnummer (beginnt bei 1).

      Rückgabe
      -------
      bool
          True, wenn Schwarz am Zug ist (ungerade Zugnummer), False sonst (gerade Zugnummer).
      """
      return zug_nummer % 2 == 1
        
    def __ergebnis(self, stellung, zug_nummer):
        """
        Berechnet das Endergebnis der Partie basierend auf der finalen Stellung
        nach WOF-Regeln.
        
        Parameter
        ----------
        stellung : spiellogik.Stellung
            Die finale Stellung des Spielbretts.
        zug_nummer : int
            Die Anzahl der tatsächlich gespielten Züge in der Partie.

        Rückgabe
        -------
        tuple of (int, int)
            Ein Tupel, das die Punkte für Schwarz und Weiß enthält
            `(punkte_schwarz, punkte_weiss)`.
        """
        steindifferenz = stellung.sum()
        # Die Steindifferenze besagt, wie viele Steine der Spieler, der am Zug 
        # ist, mehr hat als sein Gegenspieler.
        if steindifferenz == 0:
            return (32, 32)
        anzahl_leere_felder = ANZAHL_FELDER - np.count_nonzero(stellung)
        # Leere Felder werden als Punkte für den Spieler gewertet, der mehr
        # Steine hat.
        if steindifferenz > 0:
            punktdifferenz = steindifferenz + anzahl_leere_felder
        else:
            punktdifferenz = steindifferenz - anzahl_leere_felder
        # Wenn ungerade viele Züge gespielt wurden, ist Weiß am Zug.
        # Die Steindifferenz besagt dann, wie viele Steine Weiß mehr hat als 
        # Schwarz. Aus Sicht von Schwarz muss die Punktdifferenz dann 
        # umgedreht werden.
        if zug_nummer % 2 == 1: 
            punktdifferenz = -1*punktdifferenz

        return ((ANZAHL_FELDER + punktdifferenz)//2, (ANZAHL_FELDER - punktdifferenz)//2)
