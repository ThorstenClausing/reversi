import numpy as np
import torch
from abc import ABC, abstractmethod
from spiellogik import BRETTGROESSE

ANZAHL_FELDER = BRETTGROESSE**2

class Spieler(ABC):
  """
  Abstrakte Basisklasse für Spieler in einem Brettspiel.

  Diese Klasse definiert die gemeinsame Schnittstelle für alle Spieler-Implementierungen.
  Konkrete Spieler müssen die Methode `zug_waehlen` implementieren.
  """
  @abstractmethod
  def zug_waehlen(self, stellung):
    """
    Wählt einen Zug basierend auf der aktuellen Spielstellung.

    Diese Methode muss von allen konkreten Spieler-Klassen implementiert werden.

    Parameter
    ----------
    stellung : spiellogik.Stellung
        Die aktuelle Spielstellung. 

    Rückgabe
    -------
    object oder None
        Der gewählte Zug oder None, falls keine Züge möglich sind.        
    """
    pass

class Stochastischer_Spieler(Spieler):
  """
  Implementierung eines Spielers, der zufällig einen der möglichen Züge  wählt.
  """
  
  def __init__(self):
    """
    Initialisiert den stochastischen Spieler.

    Dabei wird ein NumPy-Zufallsgenerator für die Zugauswahl initialisiert.
    """
    super().__init__()
    self.rng = np.random.default_rng()

  def zug_waehlen(self, stellung):
    """
    Wählt einen zufälligen Zug aus der Liste der möglichen Züge.

    Parameter
    ----------
    stellung : object
        Die aktuelle Spielstellung.

    Rückgabe
    -------
    object oder None
        Ein zufällig ausgewählter Zug aus der Liste der möglichen Züge
        oder None, falls keine Züge möglich sind.
    """
    liste_moegliche_zuege = stellung.moegliche_zuege()
    if not liste_moegliche_zuege:
      return None
    n = self.rng.integers(len(liste_moegliche_zuege))
    return liste_moegliche_zuege[n]

class Lernender_Spieler_epsilon(Spieler):
  """
  Implementierung eines lernenden Spielers, der eine Epsilon-Greedy-Strategie verfolgt.

  Der Spieler nutzt einen Erfahrungsspeicher, um Stellungen zu bewerten und
  führt entweder einen zufälligen Zug (Exploration) oder den besten
  bewerteten Zug (Exploitation) aus, basierend auf dem Epsilon-Parameter.
  """
  
  def __init__(self, speicher, epsilon_kehrwert=10):
    """
    Initialisiert den lernenden Spieler mit Epsilon-Greedy-Strategie.

    Parameter
    ----------
    speicher : object
        Ein Objekt, das als Erfahrungsspeicher dient und Stellungen bewerten kann.
        Es wird erwartet, dass es eine Methode `bewertung_geben(stellung)` besitzt.
    epsilon_kehrwert : int, optional
        Der Kehrwert von Epsilon. Mit einer Wahrscheinlichkeit von 1/epsilon_kehrwert
        wird ein zufälliger Zug gewählt (Exploration). Standard ist 10.
    """
    super().__init__()
    self.erfahrungsspeicher = speicher
    self.epsilon_kehrwert = epsilon_kehrwert
    self.rng = np.random.default_rng()

  def epsilonkehrwert_eingeben(self, ekw):
    """
    Setzt den Kehrwert des Epsilon-Parameters.

    Parameter
    ----------
    ekw : int
        Der neue Kehrwert des Epsilon-Parameters. Muss eine positive ganze Zahl sein.
    """
    self.epsilon_kehrwert = ekw

  def zug_waehlen(self, stellung):
    """
    Wählt einen Zug basierend auf der Epsilon-Greedy-Strategie.

    Mit einer Wahrscheinlichkeit von 1/self.epsilon_kehrwert wird ein
    zufälliger Zug gewählt (Exploration). Andernfalls wird der Zug gewählt,
    der zu der am besten bewerteten Folgestellung führt (Exploitation).
    Bei Gleichstand wird einer der besten Züge zufällig ausgewählt.

    Parameter
    ----------
    stellung : object
        Die aktuelle Spielstellung.

    Rückgabe
    -------
    object oder None
        Der gewählte Zug oder None, falls keine Züge möglich sind.

    Hinweis
    -----
    Falls eine Folgestellung nicht im Erfahrungsspeicher gefunden wird (bewertung ist None),
    erhält sie eine Standardbewertung von 40.
    """
    liste_moegliche_zuege = stellung.moegliche_zuege()
    if not liste_moegliche_zuege:
        return None
    if (l := len(liste_moegliche_zuege)) == 1:
        return liste_moegliche_zuege[0]
    # Epsilon-Greedy-Entscheidung: Exploration oder Exploitation
    # 0 bedeutet Exploration (zufälligen Zug wählen)
    besten_zug_waehlen = self.rng.integers(self.epsilon_kehrwert)
    if not besten_zug_waehlen:
        # Exploration: zufälligen Zug wählen
        n = self.rng.integers(l)
        return liste_moegliche_zuege[n]
    else:
        # Exploitation: den am besten bewerteten Zug wählen
        beste_zuege = []
        beste_bewertung = 0
        for zug in liste_moegliche_zuege:
          folgestellung = stellung.copy()
          folgestellung.zug_spielen(zug)
          bewertung = self.erfahrungsspeicher.bewertung_geben(folgestellung)
          if bewertung is None:
              bewertung = 40 # Standardbewertung für unbekannte Stellungen
          if bewertung == beste_bewertung:
              beste_zuege.append(zug)
          if bewertung > beste_bewertung:
              beste_zuege = [zug]
        assert beste_zuege
        n = self.rng.integers(len(beste_zuege))
        return beste_zuege[n]


class Optimierender_Spieler(Spieler):
  """
  Implementierung eines Spielers, der immer den optimalen Zug basierend
  auf einem Erfahrungsspeicher wählt.

  Dieser Spieler verfolgt eine reine Exploitation-Strategie und versucht,
  immer den Zug zu wählen, der zur höchstbewerteten Folgestellung führt.
  """
  
  def __init__(self, speicher):
    """
    Initialisiert den optimierenden Spieler.

    Parameter
    ----------
    speicher : object
        Ein Objekt, das als Erfahrungsspeicher dient und Stellungen bewerten kann.
        Es wird erwartet, dass es eine Methode `bewertung_geben(stellung)` besitzt.
    """
    super().__init__()
    self.erfahrungsspeicher = speicher
    self.rng = np.random.default_rng()

  def zug_waehlen(self, stellung):
    """
    Wählt den Zug, der zur am besten bewerteten Folgestellung führt.

    Bei mehreren Zügen mit der gleichen besten Bewertung wird einer zufällig
    ausgewählt. Unbekannte Stellungen (bewertung ist None) werden ignoriert.
    Falls alle Folgestellungen unbekannt sind,
    wird ein zufälliger Zug aus allen möglichen Zügen gewählt.

    Parameter
    ----------
    stellung : object
        Die aktuelle Spielstellung.

    Rückgabe
    -------
    object oder None
        Der gewählte Zug oder None, falls keine Züge möglich sind.
    """
    liste_moegliche_zuege = stellung.moegliche_zuege()
    if not liste_moegliche_zuege:
      return None
    if (l := len(liste_moegliche_zuege)) == 1:
      return liste_moegliche_zuege[0]
    beste_zuege = []
    beste_bewertung = 0
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

class Lernender_Spieler_sigma(Spieler):
  """
  Implementierung eines lernenden Spielers, der Züge stochastisch
  basierend auf der Bewertung ihrer Folgestellungen wählt.

  Besser bewertete Züge haben eine höhere Wahrscheinlichkeit, gewählt zu werden 
  (Sigma-Greedy-Strategie).
  """
  
  def __init__(self, speicher):
    """
    Initialisiert den lernenden Spieler mit Sigma-Greedy-Strategie.

    Parameter
    ----------
    speicher : object
        Ein Objekt, das als Erfahrungsspeicher dient und Stellungen bewerten kann.
        Es wird erwartet, dass es eine Methode `bewertung_geben(stellung)` besitzt.
    """
    super().__init__()
    self.erfahrungsspeicher = speicher
    self.rng = np.random.default_rng()

  def zug_waehlen(self, stellung):
    """
    Wählt einen Zug stochastisch basierend auf den Bewertungen der Folgestellungen.

    Die Wahrscheinlichkeit, einen Zug zu wählen, ist proportional zur Bewertung
    der Folgestellung. Höher bewertete Stellungen werden bevorzugt (Sigma-Greedy-Strategie).

    Parameter
    ----------
    stellung : object
        Die aktuelle Spielstellung.

    Rückgabe
    -------
    object oder None
        Der gewählte Zug oder None, falls keine Züge möglich sind.

    Hinweis
    -----
    Falls eine Folgestellung nicht im Erfahrungsspeicher gefunden wird (bewertung ist None),
    erhält sie eine Standardbewertung von 40.
    Bewertungen von Null werden auf einen kleinen positiven Wert (0.01)
    gesetzt, um Probleme mit Wahrscheinlichkeiten von 0 zu vermeiden.
    """
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
    # Normalisierung der Gewichte, um eine Wahrscheinlichkeitsverteilung zu erhalten
    p = p / np.sum(p, dtype=float)
    n = self.rng.choice(l, p=p)
    return liste_moegliche_zuege[n]

class Minimax_Spieler(Spieler):
  """
  Implementierung eines Spielers, der den Minimax-Algorithmus mit Alpha-Beta-Suche verwendet,
  um den besten Zug zu finden.
  """

  def __init__(self, tiefe):
    """
    Initialisiert den Minimax-Spieler.

    Parameter
    ----------
    tiefe : int
        Die maximale Suchtiefe für den Minimax-Algorithmus.
        Entspricht der Anzahl der Züge, die vorausberechnet werden.
    """
    super().__init__()
    self.rng = np.random.default_rng()
    self.tiefe = tiefe

  def zug_waehlen(self, stellung):
    """
    Wählt einen Zug basierend auf dem Minimax-Algorithmus mit Alpha-Beta-Suche.

    Der Algorithmus durchsucht den Spielbaum bis zur `self.tiefe` und verwendet
    Alpha-Beta-Pruning, um die Suche zu optimieren. Wählt den Zug, der zum
    besten Ergebnis führt. Bei mehreren optimalen Zügen wird zufällig einer
    ausgewählt.

    Parameter
    ----------
    stellung : object
        Die aktuelle Spielstellung.

    Rückgabe
    -------
    object oder None
        Der gewählte Zug oder None, falls keine Züge möglich sind.

    Hinweis
    -----
    Die Bewertung einer Stellung erfolgt aus Sicht des Alpha_Beta_Spielers
    als Differenz der Anzahl seiner Steine und der Steine des Gegenspielers.
    """
    liste_moegliche_zuege = stellung.moegliche_zuege()
    if not liste_moegliche_zuege:
      return None
    if len(liste_moegliche_zuege) == 1:
      return liste_moegliche_zuege[0]
    bester_wert = 0  
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
    """
    Rekursive Hilfsfunktion für den Minimax-Algorithmus mit Alpha-Beta-Suche.

    Führt eine Tiefensuche im Spielbaum durch und verwendet Alpha-Beta-Pruning
    zur Optimierung.

    Parameter
    ----------
    stellung : object
        Die aktuelle Spielstellung.
    tiefe : int
        Die verbleibende Suchtiefe. Wenn 0, wird die Stellung bewertet.
    alpha : float
        Der beste (höchste) Wert, den der Maximierer (Alpha_Beta_Spieler)
        bisher entlang des aktuellen Pfades oder seiner Vorgänger gefunden hat.
    beta : float
        Der beste (niedrigste) Wert, den der Minimierer (Gegenspieler)
        bisher entlang des aktuellen Pfades oder seiner Vorgänger gefunden hat.
    gepasst : bool
        True, wenn im *vorherigen* Zug gepasst wurde.
        Wird verwendet, um zwei aufeinanderfolgende Passzüge zu erkennen, die das Spiel beenden.

    Rückgabe
    -------
    float
        Der Minimax-Wert der aktuellen Stellung aus Sicht des Maximierers.
    """
    if tiefe == 0 or np.count_nonzero(stellung) == ANZAHL_FELDER:
      # Die Bewertung einer Stellung erfolgt immer aus Sicht des Alpha_Beta_Spielers,
      # d.h. sie gibt an, wie viele Steine er mehr hat als der Gegenspieler. Der
      # Alpha_Beta_Spieler versucht also, die Bewertung zu maximieren, der Gegenspieler
      # versucht, sie zu minimieren.
      # Nach einer geraden Anzahl von Zügen (= Differenz zwischen self.tiefe und tiefe)
      # ist der Alpha_Beta_Spieler am Zug, seine Steine sind also in der Stellung
      # mit 1 markiert, die Summe aller Steine (= Matrixwerte der Stellung) gibt
      # die Bewertung daher korrekt an.
      # Nach einer ungeraden Anzahl von Zügen ist der Gegenspieler am Zug, für eine 
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
