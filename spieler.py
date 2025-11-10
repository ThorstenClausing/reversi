import numpy as np
import torch
from abc import ABC, abstractmethod
from spiellogik import BRETTGROESSE

ANZAHL_FELDER = BRETTGROESSE**2

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
      if bewertung == 0:
          bewertung = 0.01
      gewichte.append(bewertung)
    assert len(gewichte) == l
    p = np.array(gewichte)
    p = p / np.sum(p, dtype=float)
    n = self.rng.choice(l, p=p)
    return liste_moegliche_zuege[n]

class Alpha_Beta_Spieler(Spieler):

  def __init__(self, tiefe):
    super().__init__()
    self.rng = np.random.default_rng()
    self.tiefe = tiefe

  def zug_waehlen(self, stellung):
    liste_moegliche_zuege = stellung.moegliche_zuege()
    if not liste_moegliche_zuege:
      return None
    if len(liste_moegliche_zuege) == 1:
      return liste_moegliche_zuege[0]
    bester_wert = -65  
    beste_zuege = []
    for zug in liste_moegliche_zuege:
      naechste_stellung = stellung.copy()
      naechste_stellung.zug_spielen(zug)
      wert = self._minimax(
        naechste_stellung, self.tiefe - 1, -1*ANZAHL_FELDER, ANZAHL_FELDER, False)
      if wert > bester_wert:
        bester_wert = wert
        beste_zuege = [zug]
      elif wert == bester_wert:
        beste_zuege.append(zug)
    if (l := len(beste_zuege)) == 1: 
        return beste_zuege[0]
    else:
        n = self.rng.integers(l)
        return beste_zuege[n]

  def _minimax(self, stellung, tiefe, alpha, beta, gepasst):
    if tiefe == 0 or np.count_nonzero(stellung) == ANZAHL_FELDER:
      # Die Bewertung einer Stellung erfolgt immer aus Sicht des Alpha_Beta_Spielers,
      # d.h. sie gibt an, wie viele Steine er mehr hat als der Gegenspieler. Der
      # Alpha_Beta_Spieler versucht also, die Bewertung zu maximieren, der Gegenspieler
      # versucht, sie zu minimieren.
      # Nach einer geraden Anzahl von Zügen (= Differenz zwischen self.tiefe und tiefe)
      # ist der Alpha_Beta_Spieler am Zug, seine Steine sind also in der Stellung
      # mit 1 markiert, die Summe aller Steine (= Matrixwerte der Stellung) gibt
      # die Bewertung daher korrekt an.
      # Nach einer ungeraden Anzahl von Zügen ist der gegenspieler am Zug, für eine 
      # Bewertung aus Sicht des Alpha_Beta_Spielers muss die Summe aller Steine
      # daher mit -1 multipliziert werden.
      return stellung.sum() if (self.tiefe - tiefe) % 2 == 0 else -1*stellung.sum()
    liste_moegliche_zuege = stellung.moegliche_zuege()
    if not liste_moegliche_zuege:
      if gepasst:
        return stellung.sum() if (self.tiefe - tiefe) % 2 == 0 else -1*stellung.sum()
      else:
        naechste_stellung = stellung.copy()
        naechste_stellung.zug_spielen(None)
        return self._minimax(naechste_stellung, tiefe - 1, alpha, beta, True)
    if (self.tiefe - tiefe) % 2 == 0:
      # Der Alpha_Beta_Spieler (= Maximierer) ist am Zug.
      max_wert = -1*ANZAHL_FELDER
      for zug in liste_moegliche_zuege:
        naechste_stellung = stellung.copy()
        naechste_stellung.zug_spielen(zug)
        wert = self._minimax(naechste_stellung, tiefe - 1, alpha, beta, False)
        max_wert = max(max_wert, wert)
        # Alpha-Wert (= der beste Wert, den der Maximierer in den bisher ausgewerteten
        # Spielverläufen garantieren kann) aktualisieren:
        alpha = max(alpha, max_wert) 
        if beta <= alpha:
          # Alpha-Beta Pruning: Wenn der Alpha-Wert im aktuellen Teilspielbaum größer
          # ist als der Beta-Wert, den der Minimierer in den schon ausgewerteten
          # Teilspielbäumen garantieren kann, wird der Minimierer das Erreichen des
          # aktuellen Teilspielbaums verhindern. Er braucht daher nicht weiter
          # ausgewertet zu werden.
          break
      return max_wert
    else:
      # Der Gegenspieler (= Minimierer) ist am Zug.
      min_wert = ANZAHL_FELDER
      for zug in liste_moegliche_zuege:
        naechste_stellung = stellung.copy()
        naechste_stellung.zug_spielen(zug)
        wert = self._minimax(naechste_stellung, tiefe - 1, alpha, beta, False)
        min_wert = min(min_wert, wert)
        # Beta-Wert (= der beste Wert, den der Minimierer in den bisher ausgewerteten
        # Spielverläufen garantieren kann) aktualisieren:
        beta = min(beta, min_wert)
        if beta <= alpha:
          # Alpha-Beta Pruning: Wenn der Beta-Wert im aktuellen Teilspielbaum kleiner
          # ist als der Alpha-Wert, den der Maximierer in den schon ausgewerteten
          # Teilspielbäumen garantieren kann, wird der Maximierer das Erreichen des
          # aktuellen Teilspielbaums vemhindern. Er braucht daher nicht weiter
          # ausgewertet zu werden.
          break
      return min_wert
      
  
