from spiellogik import moegliche_zuege, zug_spielen

class Partieumgebung:

  def __init__(self,spieler_weiss,spieler_schwarz,auswertungsumgebung=None):
    self.spieler_weiss = spieler_weiss
    self.spieler_schwarz = spieler_schwarz
    self.auswertungsumgebung = auswertungsumgebung
    self.protokoll = []

  def starten(self,stellung=GRUNDSTELLUNG,am_zug=WEISS):
    self.stellung = stellung
    self.am_zug = am_zug
    self.zu_ende = False
    while not self.zu_ende:
      if self.am_zug == WEISS:
        zug = self.spieler_weiss.zug_waehlen(self.stellung)
      else:
        zug = self.spieler_schwarz.zug_waehlen(self.stellung)
      if zug == None: # Behandlung von Situationen ohne Zugmöglichkeit
        self.am_zug = -1*self.am_zug
        if self.am_zug == WEISS:
          zug = self.spieler_weiss.zug_waehlen(self.stellung)
        else:
          zug = self.spieler_schwarz.zug_waehlen(self.stellung)
        if zug == None:
          self.zu_ende = True
        else:
          self.protokoll.append(None)
      if not self.zu_ende:
        self.stellung = zug_spielen(self.stellung,zug,self.am_zug)
        self.protokoll.append(zug[0])
        self.am_zug = -1*self.am_zug
        if self.stellung.nonzero()[0].shape[0] == 64:
          self.zu_ende = True
    ergebnis = self.stellung.sum()
    if ergebnis > 0:
      self.auswertungsumgebung.ergebnis_speichern(1)
    elif ergebnis == 0:
      self.auswertungsumgebung.ergebnis_speichern(0.5)
    self.protokoll.append(ergebnis)
    if self.auswertungsumgebung != None:
      self.auswertungsumgebung.bewertung_aktualisieren(self.protokoll)
