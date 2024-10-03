import numpy as np
from abc import ABC, abstractmethod
from spiellogik import moegliche_zuege, zug_spielen
from auswertungsumgebung import Auswertungsumgebung

class Spieler(ABC):

  def __init__(self,farbe):
    self.farbe = farbe

  @abstractmethod
  def zug_waehlen(self,stellung):
    pass

class Stochastischer_Spieler(Spieler):

  def __init__(self,farbe):
    super().__init__(farbe)
    self.rng = np.random.default_rng()

  def zug_waehlen(self,stellung):
    liste_moegliche_zuege = moegliche_zuege(stellung,self.farbe)
    if (b := len(liste_moegliche_zuege)) == 0:
      return None
    n = self.rng.integers(b)
    return liste_moegliche_zuege[n]

class Lernender_Spieler(Spieler):

  def __init__(self,farbe,awu):
    super().__init__(farbe)
    self.auswertungsumgebung = awu
    self.rng = np.random.default_rng()

  def zug_waehlen(self,stellung):
    liste_moegliche_zuege = moegliche_zuege(stellung,self.farbe)
    if (l := len(liste_moegliche_zuege)) == 0:
      return None
    if l == 1:
      return liste_moegliche_zuege[0]
    bewertung_dict = self.auswertungsumgebung.bewertung_geben(stellung,self.farbe)
    if bewertung_dict is None:
      n = self.rng.integers(l)
      return liste_moegliche_zuege[n]
    summe = np.array([bewertung_dict[key] for key in bewertung_dict.keys()]).sum()
    n = self.rng.integers(summe)
    grenze = 0
    for zug in liste_moegliche_zuege:
      grenze += bewertung_dict[zug[0]]
      if n <= grenze:
        return zug


class Optimierender_Spieler(Spieler):

  def __init__(self,farbe,awu):
    super().__init__(farbe)
    self.auswertungsumgebung = awu
    self.rng = np.random.default_rng()

  def zug_waehlen(self,stellung):
    liste_moegliche_zuege = moegliche_zuege(stellung,self.farbe)
    if (l := len(liste_moegliche_zuege)) == 0:
      return None
    if l == 1:
      return liste_moegliche_zuege[0]
    bewertung_dict = self.auswertungsumgebung.bewertung_geben(stellung,self.farbe)
    if bewertung_dict is None:
      n = self.rng.integers(l)
      return liste_moegliche_zuege[n]
    bewertungszahl = 0
    bester_zug = None
    for zug in liste_moegliche_zuege:
      if (b := bewertung_dict[zug[0]]) > bewertungszahl:
        bester_zug = zug
        bewertungszahl = b
    return bester_zug

class Minimax_Spieler(Spieler):

  def __init__(self,farbe):
    super().__init__(farbe)

  def zug_waehlen(self,stellung):
    moegliche_zuege_eins = moegliche_zuege(stellung,self.farbe)
    if (l := len(moegliche_zuege_eins)) == 0:
      return None
    if l == 1:
      return moegliche_zuege_eins[0]
    gegner_am_zug = -1*self.farbe
    ergebnis = -65
    for zug_eins in moegliche_zuege_eins:
      stellung_eins = zug_spielen(stellung,zug_eins,self.farbe)
      moegliche_zuege_zwei = moegliche_zuege(stellung_eins,gegner_am_zug)
      if len(moegliche_zuege_zwei) == 0:
        moegliche_zuege_zwei.append(None)
      ergebnis_liste_zwei = []
      for zug_zwei in moegliche_zuege_zwei:
        if zug_zwei is not None:
          stellung_zwei = zug_spielen(stellung_eins,zug_zwei,gegner_am_zug)
        else:
          stellung_zwei = stellung_eins
        moegliche_zuege_drei = moegliche_zuege(stellung_zwei,self.farbe)
        if len(moegliche_zuege_drei) == 0:
          moegliche_zuege_drei.append(None)
        ergebnis_liste_drei = []
        for zug_drei in moegliche_zuege_drei:
          if zug_drei is not None:
            stellung_drei = zug_spielen(stellung_zwei,zug_drei,self.farbe)
          else:
            stellung_drei = stellung_zwei
          moegliche_zuege_vier = moegliche_zuege(stellung_drei,gegner_am_zug)
          if len(moegliche_zuege_vier) == 0:
            moegliche_zuege_vier.append(None)
          ergebnis_liste_vier = []
          for zug_vier in moegliche_zuege_vier:
            if zug_vier is not None:
              stellung_vier = zug_spielen(stellung_drei,zug_vier,gegner_am_zug)
            else:
              stellung_vier = stellung_drei
            ergebnis_liste_vier.append(stellung_vier.sum())
          ergebnisse_vier = gegner_am_zug*np.array(ergebnis_liste_vier)
          minimax_vier = gegner_am_zug*ergebnisse_vier.max()
          ergebnis_liste_drei.append(minimax_vier)
        ergebnisse_drei = self.farbe*np.array(ergebnis_liste_drei)
        minimax_drei = self.farbe*ergebnisse_drei.max()
        ergebnis_liste_zwei.append(minimax_drei)
      ergebnisse_zwei = gegner_am_zug*np.array(ergebnis_liste_zwei)
      minimax_zwei = gegner_am_zug*ergebnisse_zwei.max()
      if (z := self.farbe*minimax_zwei) > ergebnis:
        bester_zug = zug_eins
        ergebnis = z
    return bester_zug
