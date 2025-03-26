import pickle
import numpy as np
#from google.colab import files
from spiellogik import Stellung, als_kanonische_stellung

class Erfahrungsspeicher:

  def __init__(self, schwarz=True, weiss=False):
    self.schwarz = schwarz # Sollen Erfahrungen für Schwarz gespeichert werden?
    self.weiss = weiss     # Sollen Erfahrungen für Weiß gespeichert werden?
    self.bewertung = {}

  def speichermerkmale_setzen(self, schwarz, weiss):
    self.schwarz = schwarz
    self.weiss = weiss

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

  def __bewertung_enthaelt(self, stellung_to_bytes, anzahl_steine):
    if anzahl_steine in self.bewertung.keys():
      for b_eintrag in self.bewertung[anzahl_steine]:
        if b_eintrag['Stellung'] == stellung_to_bytes:
          return b_eintrag
    return None

  def __zu_bewertung_hinzufuegen(self, stellung_to_bytes, anzahl_steine):
    """
      Parameters
      ----------
      stellung : TYPE Stellung
          DESCRIPTION Stellung, für die ein neuer Bewertungseintrag angelegt 
          werden soll.
          Die Stellung MUSS kanonisch sein!
      anzahl_steine : TYPE Integer
          DESCRIPTION Anzahl von Steinen in der eingegebenen Stellung 
          
      Returns
      -------
      b_eintrag : TYPE Dictionary
          DESCRIPTION Bewertungseintrag für die eingegebene Stellung

      """
    b_eintrag = {'Stellung':stellung_to_bytes, 'Summe':0, 'Anzahl':0}
    if anzahl_steine in self.bewertung.keys():
      self.bewertung[anzahl_steine].append(b_eintrag)
    else:
      self.bewertung[anzahl_steine] = [b_eintrag]
    return b_eintrag

  def bewertung_aktualisieren(self, protokoll):
    stellung = Stellung()
    stellung.grundstellung()
    anzahl_steine = 4
    zug_nummer = 1
    ergebnis = protokoll.pop()//2
#    print('Aktuelles Ergebnis: ',ergebnis)
    while protokoll:
      zug = protokoll.pop(0)
      stellung.zug_spielen(zug)
      stellung_to_bytes = als_kanonische_stellung(stellung)
      if zug is not None:
          anzahl_steine += 1
      if (zug_nummer % 2 and self.schwarz) or (not zug_nummer % 2 and self.weiss):
        if not(b_eintrag := self.__bewertung_enthaelt(stellung_to_bytes, anzahl_steine)):
          b_eintrag = self.__zu_bewertung_hinzufuegen(stellung_to_bytes, anzahl_steine)
        b_eintrag['Summe'] += (ergebnis if zug_nummer % 2 else -1*ergebnis)
        b_eintrag['Anzahl'] += 1
      zug_nummer += 1      

  def bewertung_geben(self, stellung):
    anzahl_steine = np.count_nonzero(stellung)    
    if anzahl_steine in self.bewertung.keys():
      stellung_to_bytes = als_kanonische_stellung(stellung)
      for b_eintrag in self.bewertung[anzahl_steine]:        
        if b_eintrag['Stellung'] == stellung_to_bytes:
          return b_eintrag['Summe']/b_eintrag['Anzahl']
    return None

  def bewertung_drucken(self):
      print(len(self.bewertung.keys()))
      for anzahl_steine in self.bewertung.keys():
          print('\nAnzahl Steine: ',anzahl_steine, end=' - ')
          for eintrag in self.bewertung[anzahl_steine]:
              print(eintrag['Summe'],'\t',eintrag['Anzahl'], end=', ')
              
class Ergebnisspeicher:

  def __init__(self, schwarz=True, weiss=False):
    self.schwarz = schwarz # Sollen Erfahrungen für Schwarz gespeichert werden?
    self.weiss = weiss     # Sollen Erfahrungen für Weiß gespeichert werden?
    self.bewertung = {}

  def bewertung_laden(self, datei='reversi.ergebnis'):
    with (open(datei,'rb')) as f:
      self.bewertung = pickle.load(f)

  def bewertung_speichern(self, datei='reversi.ergebnis2'):
    with (open(datei,'wb')) as f:
      pickle.dump(self.bewertung,f)

  def bewertung_aktualisieren(self, protokoll):
    stellung = Stellung()
    stellung.grundstellung()
    anzahl_steine = 4
    zug_nummer = 1
    ergebnis = protokoll.pop()//2
#    print('Aktuelles Ergebnis: ',ergebnis)
    while protokoll:
      zug = protokoll.pop(0)
      stellung.zug_spielen(zug)
      stellung_to_bytes = als_kanonische_stellung(stellung)
      if zug is not None:
          anzahl_steine += 1
      if (zug_nummer % 2 and self.schwarz) or (not zug_nummer % 2 and self.weiss):
        if stellung_to_bytes not in self.bewertung.keys():
          self.bewertung[stellung_to_bytes] = (0, 0)
        summe = self.bewertung[stellung_to_bytes][0] + (ergebnis if zug_nummer % 2 else -1*ergebnis)
        anzahl = self.bewertung[stellung_to_bytes][1] + 1
        self.bewertung[stellung_to_bytes] = (summe, anzahl)
      zug_nummer += 1      

  def bewertung_geben(self, stellung):
    stellung_to_bytes = als_kanonische_stellung(stellung)
    if stellung_to_bytes in self.bewertung.keys():       
        return self.bewertung[stellung_to_bytes][0]/self.bewertung[stellung_to_bytes][1]
    return None

  def bewertung_drucken(self):
      for stellung_to_bytes in self.bewertung.keys():
          print(stellung_to_bytes, '\t', self.bewertung[stellung_to_bytes])
