"reversi" ist eine Sammlung von Modulen, um einem Computer mit reinforcement-learning-Methoden das Spiel beizubringen.

In der Datei spiellogik.py werden die Spielregeln von Reversi abgebildet. In der Datei spieler.py werden verschiedene Arten von Spielern definiert.
Die Datei partieumgebung.py stellt eine Umgebung zur Verfügung, in der zwei beliebige Spieler gegeneinander spielen können. 
Die Datei bewertungsgeber.py stellt Tabellen und Netzwerke zur Verfügung, mit denen gespeichert werden kann, welche Züge in welchen Stellungen mit welchem Endergebnis
gespielt wurden, und macht Lernenden und Optimierenden Spielern diese Daten zugänglich.
