import pickle
#from google.colab import files
from spiellogik import moegliche_zuege, zug_spielen, WEISS, SCHWARZ, GRUNDSTELLUNG

class Auswertungsumgebung:

  def __init__(self):
    self.bewertung = {}
    self.ergebnis_speicher = 0

  def bewertung_laden(self,datei='reversi.of'):
    with (open(datei,'rb')) as f:
      self.bewertung = pickle.load(f)

  def bewertung_speichern(self,datei='reversi.of'):
    with (open(datei,'wb')) as f:
      pickle.dump(self.bewertung,f)

  """
  def bewertung_speichern_colab(self,datei='reversi.of'):
    with (open(datei,'wb')) as f:
      pickle.dump(self.bewertung,f)
    files.download(datei)
  """

  def __bewertung_enthaelt(self,anzahl_steine,stellung,am_zug):
    if anzahl_steine in self.bewertung.keys():
      for b_tupel in self.bewertung[anzahl_steine]:
        if (b_tupel[0] == stellung).all() and b_tupel[1] == am_zug:
          return True
    return False

  def __zu_bewertung_hinzufuegen(self,anzahl_steine,stellung,am_zug):
    dict_stellung = {zug[0]:1 for zug in moegliche_zuege(stellung,am_zug)}
    b_tupel = (stellung,am_zug,dict_stellung)
    if anzahl_steine in self.bewertung.keys():
      self.bewertung[anzahl_steine].append(b_tupel)
    else:
      self.bewertung[anzahl_steine] = [b_tupel]

  def bewertung_aktualisieren(self,protokoll): # getrennte Bewertungen fuer weiss und schwarz?
    stellung = GRUNDSTELLUNG.copy()
    am_zug = WEISS
    anzahl_steine = 4
    ergebnis = protokoll.pop()
    while len(protokoll) > 0:
      p_zug = protokoll.pop(0)
      if len(moegliche_zuege(stellung,am_zug)) > 1:
        if not self.__bewertung_enthaelt(anzahl_steine,stellung,am_zug):
          self.__zu_bewertung_hinzufuegen(anzahl_steine,stellung,am_zug)
        if p_zug != None:
          for b_tupel in self.bewertung[anzahl_steine]:
            if (b_tupel[0] == stellung).all() and b_tupel[1] == am_zug:
              b_tupel[2][p_zug] += ergebnis
              if (saldo := b_tupel[2][p_zug]) <= 0:
                for key in b_tupel[2].keys():
                  b_tupel[2][key] += 1 - saldo
              break
      if p_zug != None:
        for zug in moegliche_zuege(stellung,am_zug):
          if zug[0] == p_zug:
            stellung = zug_spielen(stellung,zug,am_zug)
            break
        anzahl_steine += 1
      am_zug *= -1

  def bewertung_geben(self,stellung,am_zug):
    anzahl_steine = stellung.nonzero()[0].shape[0]
    if anzahl_steine in self.bewertung.keys():
      for b_tupel in self.bewertung[anzahl_steine]:
        if (stellung == b_tupel[0]).all() and am_zug == b_tupel[1]:
          return b_tupel[2]
    return None

  def ergebnis_speichern(self,ergebnis):
    self.ergebnis_speicher += ergebnis
