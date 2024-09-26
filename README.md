"reversi" ist eine Sammlung von Skripten, um einem Computer mit reinforcement-learning-Methoden beizubringen, (gut) Reversi zu spielen.

In der Datei spiellogik.py werden die Spiellregeln von Reversi in Funktionen abgebildet, die mögliche Züge und Spielstellungen beschreiben.

In der Datei spieler.py werden vier verschiedene Arten von Spielern definiert:

- Der Stochastische Spieler wählt jeweils einen der möglichen Züge gleichverteilt zufällig aus.
- Der Minimax-Spieler berechnet - mit einer Tiefe von vier Zügen - alle von der aktuellen Stellung aus erreichbaren Stellungen aus, bewertet 
diese mit der Differenz der Anzahl der weißen und schwarzen Steine, und ermittelt von dort aus rückwärts mit dem MiniMax-Algorithmus (= backward-induction-Methode) den
besten Zug für die aktuelle Stellung
- Der Lernende Spieler 
- Der Optimierende Spieler 

Technische Abhängigkeiten: Python, NumPy, Pickle, ABC
