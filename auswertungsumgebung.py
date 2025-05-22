import pickle
import numpy as np
import zipfile
#from google.colab import files
from spiellogik import Stellung, als_kanonische_stellung
             
class Ergebnisspeicher:

  def __init__(self, schwarz=True, weiss=False):
    self.schwarz = schwarz # Sollen Erfahrungen für Schwarz gespeichert werden?
    self.weiss = weiss     # Sollen Erfahrungen für Weiß gespeichert werden?
    self.bewertung = {}

  def speichermerkmale_setzen(self, schwarz, weiss):
    self.schwarz = schwarz
    self.weiss = weiss

  def bewertung_laden(self, datei='reversi_schwarz.of'):
    try:
        with zipfile.ZipFile("reversi.zip", "r") as archiv:
            with archiv.open(datei, "r") as d:
                self.bewertung = pickle.load(d)           
    except:
        print('Fehler beim Laden der Bewertung!')

  def bewertung_speichern(self, datei='reversi_schwarz.of'):
    with zipfile.ZipFile(
            "reversi.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9
            ) as archiv:
        with archiv.open(datei, "w") as d:
            pickle.dump(self.bewertung, d)

            
  def anzahl_bewertungen(self):
    return len(self.bewertung.keys())

  def bewertung_aktualisieren(self, protokoll):
    stellung = Stellung()
    stellung.grundstellung()
    zug_nummer = 0
    ergebnis = protokoll.pop()//2
#    print('Aktuelles Ergebnis: ',ergebnis)
    while protokoll:
      zug_nummer += 1
      zug = protokoll.pop(0)
      stellung.zug_spielen(zug)     
      if (zug_nummer % 2 and self.schwarz) or (not zug_nummer % 2 and self.weiss):
          stellung_to_bytes = als_kanonische_stellung(stellung)
          differenz = ergebnis if zug_nummer % 2 else -1*ergebnis
          if stellung_to_bytes not in self.bewertung.keys():
              self.bewertung[stellung_to_bytes] = (differenz, 1)
          else:
              summe = self.bewertung[stellung_to_bytes][0] + differenz
              anzahl = self.bewertung[stellung_to_bytes][1] + 1
              self.bewertung[stellung_to_bytes] = (summe, anzahl)          

  def bewertung_geben(self, stellung):
    stellung_to_bytes = als_kanonische_stellung(stellung)
    if stellung_to_bytes in self.bewertung.keys():       
        return self.bewertung[stellung_to_bytes][0]/self.bewertung[stellung_to_bytes][1]
    return None

  def bewertung_drucken(self):
      for stellung_to_bytes in self.bewertung.keys():
          #print(stellung_to_bytes, '\t', self.bewertung[stellung_to_bytes])
          print(self.bewertung[stellung_to_bytes], end='\t')

class Ergebnisspeicher_v2(Ergebnisspeicher):
    """
    Version für Ergebnisangabe im Format (ergebnis_schwarz, ergebnis_weiss)
    """
    
    def __init__(self, schwarz=True, weiss=False):
      super().__init__(schwarz, weiss)
    
    def bewertung_aktualisieren(self, protokoll):
      stellung = Stellung()
      stellung.grundstellung()
      zug_nummer = 0
      ergebnis = protokoll.pop()
  #    print('Aktuelles Ergebnis: ',ergebnis)
      while protokoll:
        zug_nummer += 1
        zug = protokoll.pop(0)
        stellung.zug_spielen(zug)     
        if (zug_nummer % 2 and self.schwarz) or (not zug_nummer % 2 and self.weiss):
            stellung_to_bytes = als_kanonische_stellung(stellung)
            inkrement = ergebnis[0] if zug_nummer % 2 else ergebnis[1]
            if stellung_to_bytes not in self.bewertung.keys():
                # Die anfängliche Bewertung sollte nicht null sein, da der Zug 
                # sonst nie wieder ausprobiert und aktualisiert wird. Daher
                # wird ein Mindestwert von 2 für die Bewertung vorgegeben.
                self.bewertung[stellung_to_bytes] = (max(2, inkrement), 1)
            else:
                summe = self.bewertung[stellung_to_bytes][0] + inkrement
                anzahl = self.bewertung[stellung_to_bytes][1] + 1
                self.bewertung[stellung_to_bytes] = (summe, anzahl)