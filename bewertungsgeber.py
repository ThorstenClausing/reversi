"""
Modul für Bewertungsmechanismen für Reversi.

Dieses Modul stellt verschiedene Klassen für die Evaluierung von Spielstellungen
zur Verfügung:
- `Bewertungstabelle`: Ein einfaches Bewertungssystem basierend auf einer Hash-Tabelle.
- `Bewertungsnetz`: Ein Feed-Forward-Neurales Netz zur Stellungsbewertung.
- `Faltendes_Bewertungsnetz`: Ein Faltungs-Neurales Netz (CNN) zur Stellungsbewertung.

Es enthält auch Hilfsklassen für die Datenverarbeitung mit PyTorch.
"""

import pickle
import zipfile
from itertools import batched
import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from spiellogik import Stellung, BRETTGROESSE, als_kanonische_stellung
from tensordict import tensorclass
             
class Bewertungstabelle:
  """
  Repräsentiert ein Hash-Tabellen-basiertes Bewertungssystem für Spielstellungen.

  Diese Klasse speichert Bewertungen für Spielstellungen in einem Wörterbuch,
  das kanonische Brettzustände auf ein Tupel von (Summe_der_Bewertungen, Anzahl_der_Bewertungen)
  abbildet. Es unterstützt das Laden, Speichern, Aktualisieren und Abrufen von Bewertungen.
  """

  def __init__(self, schwarz=True, weiss=False):
    """
    Initialisiert die Bewertungstabelle.

    Parameter
    ----------
    schwarz : bool, optional
        True, wenn Erfahrungen für Schwarz gespeichert werden sollen,
        standardmäßig True.
    weiss : bool, optional
        True, wenn Erfahrungen für Weiß gespeichert werden sollen,
        standardmäßig False.
    """
    self.schwarz = schwarz 
    self.weiss = weiss     
    self.bewertung = {}

  def speichermerkmale_setzen(self, schwarz, weiss):
    """
    Legt fest, für welche Spieler Erfahrungen gespeichert werden sollen.

    Parameter
    ----------
    schwarz : bool
        True, wenn Erfahrungen für Schwarz gespeichert werden sollen.
    weiss : bool
        True, wenn Erfahrungen für Weiß gespeichert werden sollen.
    """
    self.schwarz = schwarz
    self.weiss = weiss

  def bewertung_laden(self, dateiliste=['reversi']):
    """
    Lädt Bewertungsdaten aus einem ZIP-Archiv in das interne Bewertungs-Wörterbuch.

    Die Methode versucht, gepickelte Bewertungs-Wörterbücher aus den angegebenen
    Dateien innerhalb eines "reversi.zip"-Archivs zu laden. Falls das Laden
    fehlschlägt wird eine Fehlermeldung ausgegeben.

    Parameter
    ----------
    dateiliste : list of str, optional
        Eine Liste von Dateinamen (ohne die Erweiterung '.of'), die aus dem
        ZIP-Archiv geladen werden sollen, standardmäßig ['reversi'].

    Hinweis
    -----
    Die Dateierweiterung '.of' wird automatisch an jeden Dateinamen in `dateiliste`
    angehängt.
    """
    self.bewertung = {} 
    try:
        with zipfile.ZipFile("reversi.zip", "r") as archiv:
            for datei in dateiliste:
                with archiv.open(datei + '.of', "r") as d:
                    self.bewertung.update(pickle.load(d))
    except:
        print('Fehler beim Laden der Bewertung!')

  def bewertung_speichern(self, datei='reversi'):
      """
      Speichert die aktuellen Bewertungsdaten in einem ZIP-Archiv.

      Das Bewertungs-Wörterbuch `self.bewertung` wird gepickelt und in einem
      ZIP-Archiv namens "reversi.zip" gespeichert. Wenn das Wörterbuch sehr groß
      ist (mehr als 20 Millionen Einträge), wird es in mehrere Teile aufgeteilt,
      wobei jeder Teil als separate Datei mit einem Index im Archiv gespeichert wird.

      Parameter
      ----------
      datei : str, optional
          Der Basisdateiname für die Bewertungsdaten innerhalb des ZIP-Archivs,
          standardmäßig 'reversi'.
      """
      if len(self.bewertung) <= 20000000:
          with zipfile.ZipFile(
                "reversi.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9
                ) as archiv:
            with archiv.open(datei + '.of', "w") as d:
                pickle.dump(self.bewertung, d)
      else:
          with zipfile.ZipFile(
                "reversi.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9
                ) as archiv:
            i = 0
            for teilbewertung in batched(self.bewertung.items(), 20000000):
                with archiv.open(datei + str(i) + '.of', "w") as d:
                    pickle.dump(teilbewertung, d)
                i += 1

            
  def anzahl_bewertungen(self):
    """
    Gibt die Anzahl der aktuell in der Bewertungstabelle gespeicherten, eindeutigen Stellungen zurück.

    Rückgabe
    -------
    int
        Die Gesamtzahl der bewerteten Brettstellungen.
    """
    return len(self.bewertung.keys())

  def bewertung_geben(self, stellung):
    """
    Ruft den durchschnittlichen Bewertungswert für eine gegebene Brettstellung ab.

    Die Stellung wird zuerst in ihre kanonische Form umgewandelt. Falls die kanonische
    Stellung in der Bewertungstabelle vorhanden ist, wird ihre durchschnittliche
    Bewertung (Summe_der_Ergebnisse / Anzahl_der_Ergebnisse) zurückgegeben. Andernfalls wird 
    None zurückgegeben.

    Parameter
    ----------
    stellung : spiellogik.Stellung
        Die aktuelle Brettstellung, die bewertet werden soll.

    Rückgabewert
    -------
    float oder None
        Der durchschnittliche Bewertungswert für die Stellung, oder None,
        wenn die Stellung nicht in der Tabelle enthalten ist.
    """
    stellung_to_bytes = als_kanonische_stellung(stellung)
    if stellung_to_bytes in self.bewertung.keys():       
        return self.bewertung[stellung_to_bytes][0]/self.bewertung[stellung_to_bytes][1]
    return None

  def bewertung_drucken(self):
      """
      Gibt alle gespeicherten Bewertungen auf der Konsole aus.

      Jeder Eintrag wird als sein rohes (Summe_der_Bewertungen, Anzahl_der_Bewertungen)-Tupel
      ausgegeben, getrennt durch einen Tabulator.
      """
      for stellung_to_bytes in self.bewertung.keys():
          #print(stellung_to_bytes, '\t', self.bewertung[stellung_to_bytes])
          print(self.bewertung[stellung_to_bytes], end='\t')
    
  def bewertung_aktualisieren(self, protokoll):
    """
    Aktualisiert die Bewertungstabelle basierend auf einem abgeschlossenen Spielprotokoll.

    Die Methode durchläuft die Spielzüge im `protokoll` und aktualisiert die Bewertung
    für jede Brettstellung, die während des Spiels angetroffen wird, jedoch nur
    für den Spieler, dessen Erfahrungen gemäß Konfiguration gespeichert werden sollen
    (`self.schwarz` oder `self.weiss`). 
    Neue Positionen werden mit einem Mindest-Score initialisiert, um die Exploration
    zu fördern.

    Parameter
    ----------
    protokoll : list
        Eine Liste, die die Abfolge der Spielzüge und das Endergebnis des Spiels darstellt.
        Das letzte Element ist das `ergebnis` (Ergebnis-Tupel), und die vorhergehenden
        Elemente sind `Züge.
    """
    stellung = Stellung()
    stellung.grundstellung()
    zug_nummer = 0
    ergebnis = protokoll.pop()
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

@tensorclass
class BewertungsDaten:
    """
    Klasse zur Speicherung von Batches von Brettstellungen
    und deren Bewertungen.

    Diese Klasse ist mit `@tensorclass` von `tensordict` dekoriert, was es ihr
    ermöglicht, Sammlungen von Tensoren effizient als eine einzige Datenstruktur
    zu behandeln.
    
    Attribute
    ----------
    stellungen : torch.Tensor
        Ein Tensor, der die numerischen Darstellungen der Brettstellungen enthält.
        Shape: (batch_size, BRETTGROESSE*BRETTGROESSE) für flache Darstellungen.
    bewertungen : torch.Tensor
        Ein Tensor, der die Bewertungspunkte enthält, die jeder Stellung
        entsprechen. Shape: (batch_size, 1).
    """
    stellungen: torch.Tensor
    bewertungen: torch.Tensor

class Bewertungsnetz(nn.Module):
    """
    Ein vollvernetztes vorwärtsgerichtetes neuronales Netz zur Bewertung von Spielstellungen.

    Dieses Netz verarbeitet eine als Vektor dargestellte Stellung und gibt
    einen einzelnen Bewertungswert aus. Es enthält Optionen für die kanonische
    Transformation von Eingabestellungen und die Speicherung von Erfahrungen
    für bestimmte Spieler.
    """
  
    def __init__(self, schwarz=True, weiss=False, 
                 transformation=als_kanonische_stellung,
                 kanonisch=True,
                 zwischenspeicher=None, runden=0):
        """
        Initialisiert das Bewertungsnetz.

        Parameter
        ----------
        schwarz : bool, optional
            True, wenn Erfahrungen für Schwarz gespeichert werden sollen,
            standardmäßig True.
        weiss : bool, optional
            True, wenn Erfahrungen für Weiß gespeichert werden sollen,
            standardmäßig False.
        transformation : callable, optional
            Eine Funktion, die ein `Stellung`-Objekt in eine numerische
            Darstellung umwandelt, bevor es gespeichert wird. Standardmäßig
            `als_kanonische_stellung`, die eine bytes-ähnliche Darstellung zurückgibt.
            Diese Funktion muss eine numerische Darstellung liefern, die zu einem
            `torch.Tensor` konvertiert werden kann, wenn `schwarz` oder `weiss` True ist.
        kanonisch : bool, optional
            Wenn True, werden Stellungen in `bewertung_geben` vor der
            Übergabe an das Netz in ihre kanonische Form umgewandelt,
            standardmäßig True.
        zwischenspeicher : tensordict.TensorDict oder Ähnliches, optional
            Ein Replay Buffer oder eine ähnliche Datenstruktur zum
            Speichern von `BewertungsDaten` für das Training, standardmäßig None.
        runden : int, optional
            Anzahl der Nachkommastellen, auf die die Ausgabebewertung gerundet
            werden soll. 0 bedeutet keine Rundung, standardmäßig 0.
        """
        super(Bewertungsnetz, self).__init__()
        self.innere_schicht_eins = nn.Linear(64, 96)
        self.innere_schicht_zwei = nn.Linear(96, 34)
        self.ausgabeschicht = nn.Linear(34, 1)
        self.aktivierung_eins = nn.Tanh()
        self.aktivierung_zwei = nn.Tanh()
        self.flatten = nn.Flatten()
        nn.init.xavier_uniform_(
            self.innere_schicht_eins.weight)
        nn.init.xavier_uniform_(
            self.innere_schicht_zwei.weight)
        nn.init.xavier_uniform_(
            self.ausgabeschicht.weight)
        nn.init.zeros_(self.innere_schicht_eins.bias)
        nn.init.zeros_(self.innere_schicht_zwei.bias)
        nn.init.zeros_(self.ausgabeschicht.bias)
        self.schwarz = schwarz # Sollen Erfahrungen für Schwarz gespeichert werden?
        self.weiss = weiss     # Sollen Erfahrungen für Weiß gespeichert werden?
        self.transformation = transformation # Wie sollen Stellungen vor Speicherung transformiert werden?
        self.kanonisch = kanonisch # Sollen Stellungen vor Bewertung kanonisiert werden?
        self.zwischenspeicher = zwischenspeicher
        self.runden = runden 

    def forward(self, x):
        """
        Definiert den Forward-Pass des Neuronalen Netzes.

        Parameter
        ----------
        x : torch.Tensor
            Eingabetensor, der einen Batch von als Vektor dargestellten Brettstellungen
            repräsentiert. Erwartete Shape: (batch_size, BRETTGROESSE*BRETTGROESSE).

        Rückgabe
        -------
        torch.Tensor
            Ausgabetensor, der die Bewertungswerte für jede Eingabestellung
            enthält. Shape: (batch_size, 1).
        """
        z = self.flatten(x)
        z = self.innere_schicht_eins(z)
        z = self.aktivierung_eins(z)
        z = self.innere_schicht_zwei(z)
        z = self.aktivierung_zwei(z)
        bewertung = self.ausgabeschicht(z)
        return bewertung
    
    def speichermerkmale_setzen(self, schwarz, weiss):
        """
        Legt fest, für welche Spieler Erfahrungen gespeichert werden sollen.

        Parameter
        ----------
        schwarz : bool
            True, wenn Erfahrungen für Schwarz gespeichert werden sollen.
        weiss : bool
            True, wenn Erfahrungen für Weiß gespeichert werden sollen.
        """
        self.schwarz = schwarz
        self.weiss = weiss
        
    def rundungsparameter_setzen(self, runden):
        """
        Setzt den Rundungsparameter für die Ausgabebewertungen des Netzes.

        Parameter
        ----------
        runden : int
            Anzahl der Nachkommastellen, auf die die Bewertung gerundet werden soll.
            0 bedeutet keine Rundung.
        """
        self.runden = runden
    
    def bewertung_geben(self, stellung):
        """
        Ermittelt mit Hilfe des neuronalen Netzes eine Bewertung für eine eingegebene Stellung.

        Die Eingabestellung wird optional in ihre kanonische Form umgewandelt und
        dann vom Netz verarbeitet. Negative Ausgaben eines untrainierten Netzes
        werden auf 0 begrenzt. Die Bewertung kann gerundet werden.

        Parameter
        ----------
        stellung : spiellogik.Stellung 
            Die aktuelle Brettstellung, die bewertet werden soll. 
            
        Rückgabe
        -------
        float
            Der Bewertungswert für die Stellung.
        """
        if self.kanonisch:
            stellung = als_kanonische_stellung(stellung)
            stellung = np.frombuffer(stellung, dtype=np.int8)
        # eingabe = (torch.from_numpy(np.array([stellung]))).to(device, torch.float32)
        with torch.inference_mode():
            eingabe = torch.tensor(
                stellung, dtype=torch.float32, device=self.prozessor).unsqueeze(0)
            ausgabe = self.forward(eingabe).item()
        # Bei untrainiertem Netz sind negative Ausgaben möglich, mit denen die 
        # Spieler nicht umgehen können und die daher abgefangen werden
        # müssen:
        ausgabe = max(0, ausgabe)  
        if self.runden:
            return round(ausgabe, self.runden)
        return ausgabe
    
    def bewertung_aktualisieren(self, protokoll):
      """
      Verarbeitet ein Spielprotokoll und fügt relevante Brettstellungen und Ergebnisse
      dem `zwischenspeicher` hinzu.

      Diese Methode rekonstruiert den Spielzustand aus dem Protokoll. Für jeden Zug,
      bei dem die Erfahrungen des aktuellen Spielers gespeichert werden sollen,
      wandelt sie die Brettstellung und das entsprechende Ergebnis in ein
      `BewertungsDaten`-Objekt um, das dann dem `zwischenspeicher` hinzugefügt wird.

      Parameter
      ----------
      protokoll : list
          Eine Liste, die die Abfolge der Spielzüge und das Endergebnis des Spiels darstellt.
          Das letzte Element ist das `ergebnis` (Ergebnis-Tupel), und die vorhergehenden
          Elemente sind Züge.
      """
      stellung = Stellung()
      stellung.grundstellung()
      zug_nummer = 0
      ergebnis = protokoll.pop()
      liste_stellungen = []
      liste_bewertungen = []
      while protokoll:
          zug_nummer += 1
          zug = protokoll.pop(0)
          stellung.zug_spielen(zug)
          if (zug_nummer % 2 and self.schwarz) or (not zug_nummer % 2 and self.weiss):
              if self.transformation:
                  stellung_neu = self.transformation(stellung)
                  stellung_neu = np.frombuffer(stellung_neu, dtype=np.int8)
              else:
                  stellung_neu = stellung.copy()
              liste_stellungen.append(stellung_neu)
              bewertung = ergebnis[0] if zug_nummer % 2 else ergebnis[1]
              liste_bewertungen.append([bewertung])
      data = BewertungsDaten(
              stellungen=torch.tensor(
                  np.array(liste_stellungen), 
                  dtype=torch.float32, 
                  device=self.prozessor),
              bewertungen=torch.tensor(
                  np.array(liste_bewertungen), 
                  dtype=torch.float32, 
                  device=self.prozessor), 
              batch_size=[len(liste_stellungen)])
      self.zwischenspeicher.extend(data)
      
class Bewertungsdatensatz(Dataset):
    """
    Ein PyTorch-Dataset zur Bereitstellung von Stellungen und deren Bewertungen.

    Dieses Dataset kapselt eine Liste von (Stellung, Bewertung)-Tupeln
    und macht sie für einen PyTorch-DataLoader zugänglich.
    """
    def __init__(self, liste):
        """
        Initialisiert den Bewertungsdatensatz.

        Parameter
        ----------
        liste : list of tuple
            Eine Liste mit (Stellung, Bewertung)-Tupeln.
        """
        self.liste = liste

    def __len__(self):
        """
        Gibt die Gesamtzahl der bewerteten Stellungen im Datensatz zurück.

        Rückgabe
        -------
        int
            Die Anzahl der bewerteten Stellungen.
        """
        return len(self.liste)

    def __getitem__(self, idx):
        """
        Ruft eine bewertete Stellung aus dem Datensatz ab.

        Parameter
        ----------
        idx : int
            Der Index des abzurufenden Elements.

        Returns
        -------
        tuple
            Ein (Stellung, Bewertung)-Tupel.
        """
        return self.liste[idx][0], self.liste[idx][1]

class Faltendes_Bewertungsnetz(nn.Module):
    """
    Ein faltendes neuronales Netz (CNN) zur Bewertung von Spielstellungen.

    Dieses Netz verarbeitet eine 3-Kanal-Brettdarstellung (Steine des Gegners,
    Steine des Spielers, leere Felder) und gibt einen einzelnen Bewertungswert aus.
    """
    def __init__(self, kanonisch=True, runden=0):
        """
        Initialisiert das faltende Bewertungsnetz.

        Parameter
        ----------
        kanonisch : bool, optional
            Wenn True, werden Stellungen in `bewertung_geben` vor der
            Übergabe an das Netz in ihre kanonische Form umgewandelt,
            standardmäßig True.
        runden : int, optional
            Anzahl der Nachkommastellen, auf die die Ausgabebewertung gerundet
            werden soll. 0 bedeutet keine Rundung, standardmäßig 0.
        """
        super(Faltendes_Bewertungsnetz, self).__init__()
        # Eingabekanäle: 3 (Gegner, aktueller Spieler, leer)
        # Ausgabekanäle für die erste Faltungsschicht: 9 (groups=3 bedeutet 3 unabhängige Gruppen von je 3 Kanälen)
        self.innere_schicht_eins = nn.Conv2d(3, 9, kernel_size=3, padding=1, groups=3)
        self.innere_schicht_zwei = nn.Conv2d(9, 9, kernel_size=3, padding=1, groups=3)
        # Flachgedrückte Ausgabe der letzten Faltungsschicht: 9 Kanäle * BRETTGROESSE * BRETTGROESSE
        self.innere_schicht_drei = nn.Linear(9*BRETTGROESSE*BRETTGROESSE, 300)
        self.ausgabeschicht = nn.Linear(300, 1)
        self.aktivierung_drei = nn.Tanh()
        self.flatten = nn.Flatten()
        nn.init.xavier_uniform_(self.innere_schicht_eins.weight)
        nn.init.xavier_uniform_(self.innere_schicht_zwei.weight)
        nn.init.xavier_uniform_(self.innere_schicht_drei.weight)
        nn.init.xavier_uniform_(self.ausgabeschicht.weight)
        nn.init.zeros_(self.innere_schicht_eins.bias)
        nn.init.zeros_(self.innere_schicht_zwei.bias)
        nn.init.zeros_(self.innere_schicht_drei.bias)
        nn.init.zeros_(self.ausgabeschicht.bias)
        self.kanonisch = kanonisch
        self.runden = runden 

    def forward(self, x):
        """
        Definiert den Forward-Pass des faltenden neuronalen Netzes.

        Parameter
        ----------
        x : torch.Tensor
            Eingabetensor, der einen Batch von 3-Kanal-Brettstellungen repräsentiert.
            Erwartete Shape: (batch_size, 3, BRETTGROESSE, BRETTGROESSE).
            
        Rückgabe
        -------
        torch.Tensor
            Ausgabetensor, der die Bewertungswerte für jede Eingabestellung
            enthält. Shape: (batch_size, 1).
        """
        z = self.innere_schicht_eins(x)
        #z = self.aktivierung_eins(z)
        z = self.innere_schicht_zwei(z)
        #z = self.aktivierung_zwei(z)
        z = self.flatten(z)
        z = self.innere_schicht_drei(z)
        z = self.aktivierung_drei(z)
        bewertung = self.ausgabeschicht(z)
        return bewertung
    
    def bewertung_geben(self, stellung):
        """
        Ermittelt mit Hilfe des neuronalen Netzes eine Bewertung für eine eingegebene Stellung.

        Die Eingabestellung wird optional in ihre kanonische Form umgewandelt und
        dann in eine 3-Kanal-Darstellung transformiert. 
        Diese 3-Kanal-Darstellung wird anschließend vom Netz verarbeitet.
        Negative Ausgaben eines untrainierten Netzes werden auf 0 begrenzt.
        Der endgültige Score kann gerundet werden.

        Parameter
        ----------
        stellung : spiellogik.Stellung 
            Die aktuelle Brettstellung, die bewertet werden soll. 
            
        Rückgabe
        -------
        float
            Der Bewertungswert für die Stellung.
        """
        if self.kanonisch:
            stellung = als_kanonische_stellung(stellung)
            stellung = np.frombuffer(stellung, dtype=np.int8).reshape(BRETTGROESSE, BRETTGROESSE)
        stellung_plus = np.maximum(stellung, 0)
        stellung_minus = np.maximum(-1*stellung, 0)
        stellung_leer = 1 - stellung_plus - stellung_minus
        stellung_drei_kanaele = np.array([stellung_plus, stellung_minus, stellung_leer])      
        eingabe = (torch.tensor([stellung_drei_kanaele,])).to(torch.float32)
        ausgabe = self.forward(eingabe).item()
        # Bei untrainiertem Netz sind negative Ausgaben möglich, mit denen die 
        # Spieler nicht umgehen können und die daher abgefangen werden
        # müssen:
        ausgabe = max(0, ausgabe)  
        if self.runden:
            return round(ausgabe, self.runden)
        return ausgabe
