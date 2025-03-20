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
    b_eintrag = {'Stellung':stellung.copy(), 'Summe':0, 'Anzahl':0}
    if anzahl_steine in self.bewertung.keys():
      self.bewertung[anzahl_steine].append(b_eintrag)
    else:
      self.bewertung[anzahl_steine] = [b_eintrag]
    print('Hinzugefügt!')

  def bewertung_aktualisieren(self, protokoll):
    stellung = Stellung()
    stellung.grundstellung()
    anzahl_steine = 4
    zug_nummer = 1
    ergebnis = protokoll.pop()
    print('Aktuelles Ergebnis: ',ergebnis)
    while protokoll:
      zug = protokoll.pop(0)
      stellung.zug_spielen(zug)
      if zug is not None:
          anzahl_steine += 1
      if (zug_nummer % 2 and self.schwarz) or (not zug_nummer % 2 and self.weiss):
        if not self.__bewertung_enthaelt(stellung, anzahl_steine):
          self.__zu_bewertung_hinzufuegen(stellung, anzahl_steine)
        for b_eintrag in self.bewertung[anzahl_steine]:
          if (b_eintrag['Stellung'] == stellung).all():
              b_eintrag['Summe'] += (ergebnis if zug_nummer % 2 else -1*ergebnis)
              b_eintrag['Anzahl'] += 1
              break
      zug_nummer += 1
      

  def bewertung_geben(self, stellung):
    anzahl_steine = stellung.nonzero()[0].shape[0]
    if anzahl_steine in self.bewertung.keys():
      for b_eintrag in self.bewertung[anzahl_steine]:
        if (stellung == b_eintrag['Stellung']).all():
          return b_eintrag['Summe'], b_eintrag['Anzahl']
    return None

  def bewertung_drucken(self):
      print(len(self.bewertung.keys()))
      for anzahl_steine in self.bewertung.keys():
          print('Anzahl Steine: ',anzahl_steine)
          print(self.bewertung[anzahl_steine])
