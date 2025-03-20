import pickle
import numpy as np
#from google.colab import files
from spiellogik import Stellung

class Erfahrungsspeicher:

  def __init__(self, schwarz=True, weiss=False):
    self.schwarz = schwarz # Sollen Erfahrungen für Schwarz gespeichert werden?
    self.weiss = weiss     # Sollen Erfahrungen für Weiß gespeichert werden?
    self.bewertung = {}
#    self.ergebnis_speicher = 0

  def bewertung_laden(self, datei='reversi.of'):
    with (open(datei,'rb')) as f:
      self.bewertung = pickle.load(f)

  def bewertung_speichern(self, datei='reversi.of'):
    with (open(datei,'wb')) as f:
      pickle.dump(self.bewertung,f)

  """
  def bewertung_speichern_colab(self,datei='reversi.of'):
    with (open(datei,'wb')) as f:
      pickle.dump(self.bewertung,f)
    files.download(datei)
  """

  def __bewertung_enthaelt(self, stellung, anzahl_steine=None):
    if anzahl_steine is None: anzahl_steine = np.count_nonzero(stellung) 
    if anzahl_steine in self.bewertung.keys():
      for b_tupel in self.bewertung[anzahl_steine]:
        if (b_tupel[0] == stellung).all():
          return True
    return False

  def __zu_bewertung_hinzufuegen(self, stellung, anzahl_steine=None):
    if anzahl_steine is None: anzahl_steine = np.count_nonzero(stellung) 
    b_tupel = (stellung, 0, 0)
    if anzahl_steine in self.bewertung.keys():
      self.bewertung[anzahl_steine].append(b_tupel)
    else:
      self.bewertung[anzahl_steine] = [b_tupel]

  def bewertung_aktualisieren(self, protokoll):
    stellung = Stellung()
    stellung.grundstellung()
    anzahl_steine = 4
    zug_nummer = 1
    ergebnis = protokoll.pop()
    while protokoll:
      zug = protokoll.pop(0)
      stellung.zug_spielen(zug)
      if zug is not None:
          anzahl_steine += 1
      if (zug_nummer % 2 and self.schwarz) or (not zug_nummer % 2 and self.weiss):
        if not self.__bewertung_enthaelt(anzahl_steine, stellung):
          self.__zu_bewertung_hinzufuegen(anzahl_steine, stellung)
      for b_tupel in self.bewertung[anzahl_steine]:
        if (b_tupel[0] == stellung).all():
          b_tupel[1] += ergebnis if zug_nummer % 2 else -1*ergebnis
          b_tupel[2] += 1
          break
      zug_nummer += 1
      

  def bewertung_geben(self, stellung):
    anzahl_steine = stellung.nonzero()[0].shape[0]
    if anzahl_steine in self.bewertung.keys():
      for b_tupel in self.bewertung[anzahl_steine]:
        if (stellung == b_tupel[0]).all():
          return b_tupel[1], b_tupel[2]
    return None

#  def ergebnis_speichern(self, ergebnis):
#    self.ergebnis_speicher += ergebnis
