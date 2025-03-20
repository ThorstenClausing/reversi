import numpy as np
from abc import ABC, abstractmethod

class Spieler(ABC):

  @abstractmethod
  def zug_waehlen(self, stellung):
    pass

class Stochastischer_Spieler(Spieler):

  def __init__(self):
    super().__init__()
    self.rng = np.random.default_rng()

  def zug_waehlen(self, stellung):
    liste_moegliche_zuege = stellung.moegliche_zuege()
    if not liste_moegliche_zuege:
      return None
    n = self.rng.integers(len(liste_moegliche_zuege))
    return liste_moegliche_zuege[n]

class Lernender_Spieler(Spieler):

  def __init__(self, speicher):
    super().__init__()
    self.erfahrungsspeicher = speicher
    self.rng = np.random.default_rng()

  def zug_waehlen(self, stellung):
    liste_moegliche_zuege = stellung.moegliche_zuege()
    if not liste_moegliche_zuege:
      return None
    if (l := len(liste_moegliche_zuege)) == 1:
      return liste_moegliche_zuege[0]
    bewertungen = []
    for zug in liste_moegliche_zuege:
      folgestellung = stellung.copy()
      folgestellung.zug_spielen(zug)
      erfahrung = self.erfahrungsspeicher.bewertung_geben(folgestellung)
      if erfahrung is None:
          bewertung = 1
      else:
          bewertung = erfahrung[0]/erfahrung[1]
      bewertungen.append(bewertung)
    wahrscheinlichkeiten = np.array(bewertungen)
    wahrscheinlichkeiten = wahrscheinlichkeiten / np.sum(wahrscheinlichkeiten)
    n = self.rng.choice(l, p=wahrscheinlichkeiten)
    return liste_moegliche_zuege[n]

class Optimierender_Spieler(Spieler):

  def __init__(self, speicher):
    super().__init__()
    self.erfahrungsspeicher = speicher
    self.rng = np.random.default_rng()

  def zug_waehlen(self, stellung):
    liste_moegliche_zuege = stellung.moegliche_zuege()
    if not liste_moegliche_zuege:
      return None
    if (l := len(liste_moegliche_zuege)) == 1:
      return liste_moegliche_zuege[0]
    bewertung_dict = self.erfahrungsspeicher.bewertung_geben(stellung)
    if bewertung_dict is None:
      n = self.rng.integers(l)
      return liste_moegliche_zuege[n]
    bewertungszahl = -65
    bester_zug = None
    for zug in liste_moegliche_zuege:
      if (b := bewertung_dict[zug[0]]) > bewertungszahl:
        bester_zug = zug
        bewertungszahl = b
    return bester_zug

class Minimax_Spieler(Spieler): #Prüfen!!

  def zug_waehlen(self, stellung):
    moegliche_zuege_eins = stellung.moegliche_zuege()
    if not moegliche_zuege_eins:
      return None
    if len(moegliche_zuege_eins) == 1:
      return moegliche_zuege_eins[0]
    ergebnis = -65
    for zug_eins in moegliche_zuege_eins:
      stellung_eins = stellung.copy().zug_spielen(zug_eins)
      moegliche_zuege_zwei = stellung_eins.moegliche_zuege()
      if not moegliche_zuege_zwei:
        moegliche_zuege_zwei.append(None)
      ergebnis_liste_zwei = []
      for zug_zwei in moegliche_zuege_zwei:
        if zug_zwei is not None:
          stellung_zwei = stellung_eins.copy().zug_spielen(zug_zwei)
        else:
          stellung_zwei = stellung_eins
        moegliche_zuege_drei = stellung_zwei.moegliche_zuege()
        if not moegliche_zuege_drei:
          moegliche_zuege_drei.append(None)
        ergebnis_liste_drei = []
        for zug_drei in moegliche_zuege_drei:
          if zug_drei is not None:
            stellung_drei = stellung_zwei.copy().zug_spielen(zug_drei)
          else:
            stellung_drei = stellung_zwei
          moegliche_zuege_vier = stellung_drei.moegliche_zuege()
          if not moegliche_zuege_vier:
            moegliche_zuege_vier.append(None)
          ergebnis_liste_vier = []
          for zug_vier in moegliche_zuege_vier:
            if zug_vier is not None:
              stellung_vier = stellung_drei.copy().zug_spielen(zug_vier)
            else:
              stellung_vier = stellung_drei
            ergebnis_liste_vier.append(stellung_vier.sum())
          ergebnisse_vier = np.array(ergebnis_liste_vier)
          minimax_vier = ergebnisse_vier.min()
          ergebnis_liste_drei.append(minimax_vier)
        ergebnisse_drei = np.array(ergebnis_liste_drei)
        minimax_drei = ergebnisse_drei.max()
        ergebnis_liste_zwei.append(minimax_drei)
      ergebnisse_zwei = np.array(ergebnis_liste_zwei)
      minimax_zwei = ergebnisse_zwei.min()
      if minimax_zwei > ergebnis:
        bester_zug = zug_eins
        ergebnis = minimax_zwei
    return bester_zug
