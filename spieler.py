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

class Lernender_Spieler_epsilon(Spieler):

  def __init__(self, speicher, epsilon_kehrwert=10):
    super().__init__()
    self.erfahrungsspeicher = speicher
    self.epsilon_kehrwert = epsilon_kehrwert
    self.rng = np.random.default_rng()

  def epsilonkehrwert_eingeben(self, ekw):
    self.epsilon_kehrwert = ekw

  def zug_waehlen(self, stellung):
    liste_moegliche_zuege = stellung.moegliche_zuege()
    if not liste_moegliche_zuege:
        return None
    if (l := len(liste_moegliche_zuege)) == 1:
        return liste_moegliche_zuege[0]
    besten_zug_waehlen = self.rng.integers(self.epsilon_kehrwert)
    if not besten_zug_waehlen:
        n = self.rng.integers(l)
        return liste_moegliche_zuege[n]
    else:
        beste_zuege = []
        beste_bewertung = 0
        for zug in liste_moegliche_zuege:
          folgestellung = stellung.copy()
          folgestellung.zug_spielen(zug)
          bewertung = self.erfahrungsspeicher.bewertung_geben(folgestellung)
          if bewertung is None:
              bewertung = 40
          if bewertung == beste_bewertung:
              beste_zuege.append(zug)
          if bewertung > beste_bewertung:
              beste_zuege = [zug]
        assert beste_zuege
        n = self.rng.integers(len(beste_zuege))
        return beste_zuege[n]


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
    beste_zuege = []
    beste_bewertung = -65
    for zug in liste_moegliche_zuege:
      folgestellung = stellung.copy()
      folgestellung.zug_spielen(zug)
      bewertung = self.erfahrungsspeicher.bewertung_geben(folgestellung)
      if bewertung is None:
          continue
      if bewertung == beste_bewertung:
          beste_zuege.append(zug)
      if bewertung > beste_bewertung:
          beste_zuege = [zug]
          beste_bewertung = bewertung
    if beste_zuege:
        n = self.rng.integers(len(beste_zuege))
        return beste_zuege[n]
    else:
        n = self.rng.integers(l)
        return liste_moegliche_zuege[n]

class Minimax_Spieler(Spieler):
    
  def __init__(self):
    super().__init__()
    self.rng = np.random.default_rng()

  def zug_waehlen(self, stellung):
    moegliche_zuege_eins = stellung.moegliche_zuege()
    if not moegliche_zuege_eins:
      return None
    if len(moegliche_zuege_eins) == 1:
      return moegliche_zuege_eins[0]
    ergebnis = -65
    for zug_eins in moegliche_zuege_eins:
      stellung_eins = stellung.copy()
      stellung_eins.zug_spielen(zug_eins)
      moegliche_zuege_zwei = stellung_eins.moegliche_zuege()
      if not moegliche_zuege_zwei:
        moegliche_zuege_zwei.append(None)
      ergebnis_liste_zwei = []
      for zug_zwei in moegliche_zuege_zwei:
        stellung_zwei = stellung_eins.copy()
        stellung_zwei.zug_spielen(zug_zwei)
        moegliche_zuege_drei = stellung_zwei.moegliche_zuege()
        if not moegliche_zuege_drei:
          moegliche_zuege_drei.append(None)
        ergebnis_liste_drei = []
        for zug_drei in moegliche_zuege_drei:
          stellung_drei = stellung_zwei.copy()
          stellung_drei.zug_spielen(zug_drei)
          moegliche_zuege_vier = stellung_drei.moegliche_zuege()
          if not moegliche_zuege_vier:
            moegliche_zuege_vier.append(None)
          ergebnis_liste_vier = []
          for zug_vier in moegliche_zuege_vier:
            stellung_vier = stellung_drei.copy()
            stellung_vier.zug_spielen(zug_vier)
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
        bester_zug = [zug_eins]
        ergebnis = minimax_zwei
      elif minimax_zwei == ergebnis:
        bester_zug.append(zug_eins)
    if (l := len(bester_zug)) == 1: 
        return bester_zug[0]
    else:
        n = self.rng.integers(l)
        return bester_zug[n]

class Lernender_Spieler_sigma(Spieler):

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
    gewichte = []
    for zug in liste_moegliche_zuege:
      folgestellung = stellung.copy()
      folgestellung.zug_spielen(zug)
      bewertung = self.erfahrungsspeicher.bewertung_geben(folgestellung)
      if bewertung is None:
          bewertung = 40
      gewichte.append(bewertung)
    assert len(gewichte) == l
    p = np.array(gewichte)
    p = p / np.sum(p, dtype=float)
    n = self.rng.choice(l, p=p)
    return liste_moegliche_zuege[n]