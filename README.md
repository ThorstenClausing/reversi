"reversi" ist eine Sammlung von Skripten, um einem Computer mit reinforcement-learning-Methoden beizubringen, (gut) Reversi zu spielen.

In der Datei spiellogik.py werden die Spielregeln von Reversi in Funktionen abgebildet, die mögliche Züge und Spielstellungen beschreiben.

In der Datei spieler.py werden vier verschiedene Arten von Spielern definiert:

- Der Stochastische Spieler wählt jeweils einen der möglichen Züge gleichverteilt zufällig aus.
- Der Minimax-Spieler berechnet - mit einer Tiefe von vier Zügen - alle von der aktuellen Stellung aus erreichbaren Stellungen aus, bewertet 
diese mit der Differenz der Anzahl der weißen und schwarzen Steine, und ermittelt von dort aus rückwärts mit dem MiniMax-Algorithmus (= backward-induction-Methode) den
besten Zug für die aktuelle Stellung.
- Der Lernende Spieler merkt sich für alle Stellungen, die ihm schon einmal begegnet sind, welchen Zuge er gespielt hat und zu welchem Endergebnis dies geführt hat, und wählt dann 
einen Zug mit umso höherer Wahrscheinlichkeit, je erfolgreicher sich dieser Zug in vorhergehenden Partien erwiesen hat.
- Der Optimierende Spieler wählt jeweils deterministisch denjenigen Zug, der sich in vorhergehenden Partien in der aktuellen Stellung als am erfolgreichsten erwiesen hat.

Die Datei partieumgebung.py stellt eine Umgebung zur Verfügung, in der zwei beliebige Spieler gegeneinander spielen können.

Die Datei auswertungsumgebung.py stellt eine Umgebung zur Verfügung, die für eine Serie von Partien aufzeichnet, welche Züge in welchen Stellungen mit welchem Endergebnis
gespielt wurden, und macht Lernenden und Optimierenden Spielern diese Daten zugänglich.  

Die Datei gui_umgebung.py stellt eine graphische Nutzerschnittstelle bereit [noch nicht implementiert].

Technische Abhängigkeiten: Python, NumPy, Pickle, ABC
