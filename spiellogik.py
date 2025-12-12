import numpy as np

AM_ZUG = 1
NICHT_AM_ZUG = -1
LEER = 0
BRETTGROESSE = 8

OBEN_LINKS = np.array([-1, -1], dtype=np.int8)
OBEN = np.array([-1, 0], dtype=np.int8)
OBEN_RECHTS = np.array([-1, 1], dtype=np.int8)
LINKS = np.array([0, -1], dtype=np.int8)
RECHTS = np.array([0, 1], dtype=np.int8)
UNTEN_LINKS = np.array([1, -1], dtype=np.int8)
UNTEN = np.array([1, 0], dtype=np.int8)
UNTEN_RECHTS = np.array([1, 1], dtype=np.int8)

RICHTUNGEN = [OBEN_LINKS, OBEN, OBEN_RECHTS, LINKS, RECHTS, UNTEN_LINKS, UNTEN, UNTEN_RECHTS]

class Stellung(np.ndarray):
    """
    Repräsentiert eine Reversi-Stellung.

    Diese Klasse erweitert `numpy.ndarray`, um ein 2D-Array als Spielbrett zu nutzen,
    wobei jeder Eintrag den Zustand eines Feldes (leer = LEER = 0, 
    Stein des aktuellen Spielers = AM_ZUG = 1,
    Stein des Gegenspielers = NICHT_AM_ZUG = -1) angibt.
    """
    
    def __new__(cls):
        """
        Erstellt eine neue, leere Reversi-Spielstellung.

        Diese Methode wird aufgerufen, um eine Instanz der Klasse `Stellung` zu erzeugen,
        die intern ein `numpy.ndarray` der Größe `BRETTGROESSE` x `BRETTGROESSE` ist.
        Alle Felder werden initial auf `LEER` gesetzt.

        Parameter
        ----------
        cls : type
            Die Klasse `Stellung`.

        Rückgabe
        -------
        Stellung
            Eine neue `Stellung`-Instanz, die ein leeres Reversi-Brett repräsentiert.
        """
        stellung = np.ndarray.__new__(cls, (BRETTGROESSE,BRETTGROESSE), dtype=np.int8)
        stellung.fill(LEER)
        return stellung
      
    def __array_wrap__(self, array, context=None, return_scalar=False):
        """
        Ermöglicht die Anwendung von NumPy-Operationen auf `Stellung`-Objekte. 

        Parameter
        ----------
        array : numpy.ndarray
            Das Array, das von der NumPy-Operation zurückgegeben wurde.
        context : tuple, optional
            Ein Tupel, das Informationen über die aufrufende Funktion enthält
            (hier immer None).
        return_scalar : bool, optional
            Gibt an, ob ein Skalarwert zurückgegeben werden soll (im Fall eines 0D-Arrays;
            hier immer False).

        Rückgabe
        -------
        numpy.ndarray
            Das unveränderte Array, wie es von der NumPy-Operation zurückgegeben wurde.
        """
        return array[()]
      
    def __nicht_am_zug(self, liste):
        """
        Platziert Steine des Spielers, der nicht am Zug ist, auf den angegebenen Feldern.

        Diese private Hilfsmethode ändert den Status der Felder, die in der
        `liste` von Koordinaten angegeben sind, auf `NICHT_AM_ZUG`.

        Parameter
        ----------
        liste : list of tuple
            Eine Liste von (Zeile, Spalte)-Tupeln, die die Positionen
            der zu setzenden Steine angeben.
        """
        for index in liste:
            self[index[0],index[1]] = NICHT_AM_ZUG

    def __am_zug(self, liste):
        """
        Platziert Steine des aktuellen Spielers auf den angegebenen Feldern.

        Diese private Hilfsmethode ändert den Status der Felder, die in der
        `liste` von Koordinaten angegeben sind, auf `AM_ZUG`.

        Parameter
        ----------
        liste : list of tuple
            Eine Liste von (Zeile, Spalte)-Tupeln, die die Positionen
            der zu setzenden Steine angeben.
        """
        for index in liste:
            self[index[0],index[1]] = AM_ZUG  

    def grundstellung(self):
        """
        Initialisiert das Spielbrett mit der Standard-Startstellung von Reversi.
        """
        unten  = BRETTGROESSE//2
        rechts = BRETTGROESSE//2
        oben, links = unten - 1, rechts - 1
        self.__am_zug([(oben, rechts), (unten, links)])
        self.__nicht_am_zug([(oben, links), (unten, rechts)])

    def moegliche_zuege(self):
        """
        Ermittelt alle möglichen Züge für den aktuellen Spieler.

        Ein Zug ist gültig, wenn er auf ein leeres Feld gesetzt wird und
        mindestens einen Stein des Gegners in einer beliebigen der 8 Richtungen
        einschließt, der dann umgedreht werden kann.

        Rückgabe
        -------
        list of tuple
            Eine Liste von Tupeln, wobei jedes Tupel einen möglichen Zug darstellt.
            Jeder Zug besteht aus:
            - Einem (Zeile, Spalte)-Tupel des leeren Feldes, auf das gesetzt werden kann.
            - Einer Liste von Richtungsvektoren, in denen Steine des Gegners
              eingeschlossen würden.
            Gibt eine leere Liste zurück, wenn keine Züge möglich sind.
        """
        zuege = []
        for z in range(BRETTGROESSE):
            for s in range(BRETTGROESSE):
                if self[z, s] == LEER:
                    r_liste = []
                    for richtung in RICHTUNGEN:
                        a, b = np.array([z, s]) + richtung
                        if 0 <= a < BRETTGROESSE and 0 <= b < BRETTGROESSE and self[a, b] == NICHT_AM_ZUG:
                            if self.__einschluss(a, b, richtung):
                                r_liste.append(richtung)
                    if r_liste:
                        zuege.append(((z, s), r_liste))
        return zuege

    def __einschluss(self, z, s, richtung):
        """
        Prüft, ob von einer gegebenen Position in einer bestimmten Richtung
        eine Kette von gegnerischen Steinen von einem eigenen Stein eingeschlossen wird.

        Diese Methode wird iterativ aufgerufen, um zu überprüfen,
        ob ein potenzieller Zug an der (ursprünglichen leeren) Position
        `np.array([z, s]) - richtung` in `richtung` zu einem Einschluss führt.
        Der Startpunkt `(z, s)` muss dabei der erste gegnerische Stein nach dem
        potenziellen Zugfeld sein. Die Methode prüft, ob in `richtung` eine Kette
        von gegnerischen Steinen endet, die von einem eigenen Stein abgeschlossen wird.

        Parameter
        ----------
        z : int
            Die Zeilenkoordinate des ersten gegnerischen Steins nach dem Zugfeld.
        s : int
            Die Spaltenkoordinate des ersten gegnerischen Steins nach dem Zugfeld.
        richtung : numpy.ndarray
            Der Richtungsvektor (z.B. OBEN, UNTEN_RECHTS), in dem die Prüfung erfolgt.

        Rückgabe
        -------
        bool
            True, wenn eine Kette von gegnerischen Steinen in `richtung`
            von einem Stein des aktuellen Spielers eingeschlossen wird,
            False sonst (z.B. wenn ein leeres Feld oder der Brettrand erreicht wird).

        Wirft
        ------
        AssertionError
            Wenn das Startfeld `(z, s)` nicht auf dem Brett ist oder nicht
            einen Stein des Gegenspielers enthält, obwohl dies aufgrund der
            Aufrufslogik erwartet wird.
        """
        if not (0 <= z < BRETTGROESSE and 0 <= s < BRETTGROESSE) or self[z, s] != NICHT_AM_ZUG:
            assert False
        a, b = z, s
        while True:
            c, d = np.array([a, b]) + richtung
            if not (0 <= c < BRETTGROESSE and 0 <= d < BRETTGROESSE):
                return False
            if self[c, d] == AM_ZUG:
                return True
            if self[c, d] == LEER:
                return False
            a, b = c, d

    def zug_spielen(self, zug):
        """
        Führt einen gegebenen Zug auf dem Spielbrett aus.

        Der Stein des aktuellen Spielers wird auf das angegebene leere Feld
        platziert. Alle gegnerischen Steine, die durch diesen Zug in den
        angegebenen Richtungen eingeschlossen werden, werden umgedreht. 
        Nach Ausführung des Zuges wechselt der Zug zum Gegenspieler 
        (durch Invertierung aller Feldwerte).

        Parameter
        ----------
        zug : tuple oder None
            Ein Tupel, das den Zug repräsentiert:
            - `zug[0]` : Ein (Zeile, Spalte)-Tupel des Feldes, auf das gesetzt wird.
            - `zug[1]` : Eine Liste von Richtungsvektoren, in denen Steine
              umgedreht werden müssen.
            Wenn `zug` None ist, bedeutet dies, dass der aktuelle Spieler keinen
            gültigen Zug hat und der Zug direkt an den Gegenspieler übergeht.

        Wirft
        ------
        AssertionError
            Wenn `zug` nicht None ist, aber das Zielfeld bereits belegt ist.
            Oder wenn `zug` None ist, aber tatsächlich gültige Züge möglich wären.
        """
        if zug is None:
            assert not self.moegliche_zuege()
        else:
            z, s = zug[0]
            assert self[z, s] == LEER
            self[z, s] = AM_ZUG
            for richtung in zug[1]:
                umzudrehende_steine = self.__eingeschlossene_steine(z, s, richtung)
                for stein in umzudrehende_steine:
                    self[stein[0], stein[1]] = AM_ZUG
        self.__gegenspieler_kommt_zum_zug()
            
    def __eingeschlossene_steine(self, z, s, richtung):
        """
        Ermittelt alle gegnerischen Steine in einer bestimmten Richtung,
        die durch einen neuen Stein bei `(z, s)` eingeschlossen werden.

        Diese private Hilfsmethode geht von der Position `(z, s)` in `richtung`
        über das Spielbrett und sammelt alle gegnerischen Steine, bis ein eigener Stein
        erreicht wird. Diese gesammelten Steine sind die, die umgedreht werden müssen.

        Parameter
        ----------
        z : int
            Die Zeilenkoordinate des soeben gesetzten Steins (der den Einschluss initiiert).
        s : int
            Die Spaltenkoordinate des soeben gesetzten Steins.
        richtung : numpy.ndarray
            Der Richtungsvektor, in dem über das Spielbrett gegangen werden sollen.

        Rückgabe
        -------
        list of tuple
            Eine Liste von (Zeile, Spalte)-Tupeln der Steine, die umgedreht werden müssen.

        Wirft
        ------
        AssertionError
            Wenn die erste Position nach dem Zugfeld außerhalb des Bretts liegt,
            oder wenn ein leeres Feld erreicht wird, bevor ein eigener Stein gefunden wird.
            Dies deutet auf einen internen Fehler hin, da diese Methode nur bei einem
            validierten Zug aufgerufen werden sollte.
        """
        a, b = np.array([z, s]) + richtung
        assert 0 <= a < BRETTGROESSE and 0 <= b < BRETTGROESSE
        steine = [(a, b)]
        while True:
            c, d = np.array([a, b]) + richtung
            if not (0 <= c < BRETTGROESSE and 0 <= d < BRETTGROESSE):
                assert False
            if self[c, d] == AM_ZUG:
                return steine
            if self[c, d] == LEER:
                assert False
            steine.append((c, d))
            a, b = c, d

    def __gegenspieler_kommt_zum_zug(self):
        """
        Macht den Spieler, der am Zug ist, zum Spieler, der nicht am Zug ist,
        und umgekehrt.

        Diese private Hilfsmethode invertiert den Wert aller Felder des Spielbretts.
        `AM_ZUG` (1) wird zu `NICHT_AM_ZUG` (-1) und umgekehrt.
        Leere Felder (`LEER`, 0) bleiben unverändert.
        """
        with np.nditer(self, op_flags=['readwrite'], flags=['external_loop']) as it:
            for x in it:
                x[...] = -x
        
    def stellung_anzeigen(self):
        """
        Gibt die aktuelle Spielstellung auf der Konsole aus.

        'X' repräsentiert einen Stein des aktuellen Spielers (`AM_ZUG`).
        'O' repräsentiert einen Stein des Gegenspielers (`NICHT_AM_ZUG`).
        '-' repräsentiert ein leeres Feld (`LEER`).
        """
        spalte = 0
        for feld in np.nditer(self):
            match(feld):
                case 1: print(' X', end='')
                case -1: print(' O', end='')
                case _: print(' -', end='')
            spalte += 1
            if spalte == BRETTGROESSE:
                spalte = 0
                print('\n', end='')   

def als_kanonische_stellung(stellung):
    """
    Konvertiert eine gegebene Spielstellung in ihre kanonische Form.

    Die kanonische Form ist die lexikographisch kleinste Repräsentation
    der Stellung unter allen 8 möglichen Symmetrieoperationen (Rotationen und Spiegelungen).
    
    Parameter
    ----------
    stellung : Stellung
        Die zu kanonisierende Spielstellung.

    Rückgabe
    -------
    bytes
        Die byteweise Repräsentation der kanonischen Stellung.
        Dies ist der `tobytes()`-Output des `numpy.ndarray`, der die kanonische
        Form repräsentiert.

    Hinweise
    --------
    Die 8 Symmetrieoperationen, die angewendet werden, sind:
    1. Die Originalstellung.
    2. Rotation um 90 Grad gegen den Uhrzeigersinn.
    3. Rotation um 180 Grad gegen den Uhrzeigersinn.
    4. Rotation um 270 Grad gegen den Uhrzeigersinn.
    5. Eine Transformation der 270-Grad-rotierten Stellung: Spiegelung an der Hauptdiagonale
       gefolgt von einer 90-Grad-Rotation.
    6. Horizontale Spiegelung der Originalstellung (`np.fliplr`).
    7. Vertikale Spiegelung der Originalstellung (`np.flipud`).
    8. Spiegelung der Originalstellung an der Hauptdiagonale (`np.transpose`).

    Die lexikographisch kleinste byteweise Darstellung wird durch den Vergleich
    der `tobytes()`-Repräsentationen dieser 8 symmetrischen Varianten gefunden.
    """
    stellung_eins = stellung.copy()
    stellung_to_bytes = stellung_eins.tobytes()
    
    # 90 Grad nach links rotieren
    stellung_zwei = np.rot90(stellung_eins)
    if stellung_zwei.tobytes() < stellung_to_bytes:
        stellung_to_bytes = stellung_zwei.tobytes()
        
    # 180 Grad nach links rotieren    
    stellung_zwei = np.rot90(stellung_zwei)
    if stellung_zwei.tobytes() < stellung_to_bytes:
        stellung_to_bytes = stellung_zwei.tobytes()
        
    # 270 Grad nach links rotieren
    stellung_zwei = np.rot90(stellung_zwei)
    if stellung_zwei.tobytes() < stellung_to_bytes:
        stellung_to_bytes = stellung_zwei.tobytes()
        
    # an Nebendiagonale spiegeln
    stellung_zwei = np.rot90(np.transpose(stellung_zwei))
    if stellung_zwei.tobytes() < stellung_to_bytes:
        stellung_to_bytes = stellung_zwei.tobytes()
        
    # vertikal spiegeln    
    stellung_zwei = np.fliplr(stellung_eins)
    if stellung_zwei.tobytes() < stellung_to_bytes:
        stellung_to_bytes = stellung_zwei.tobytes()
        
    # horizontal spiegeln    
    stellung_zwei = np.flipud(stellung_eins)
    if stellung_zwei.tobytes() < stellung_to_bytes:
        stellung_to_bytes = stellung_zwei.tobytes()
        
    # an Hauptdiagonale spiegeln
    stellung_zwei = np.transpose(stellung_eins)
    if stellung_zwei.tobytes() < stellung_to_bytes:
        stellung_to_bytes = stellung_zwei.tobytes()
        
    return stellung_to_bytes
