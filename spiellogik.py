import numpy as np

WEISS = 1
SCHWARZ = -1

OBEN_LINKS = np.array([-1, -1])
OBEN = np.array([-1, 0])
OBEN_RECHTS = np.array([-1, 1])
LINKS = np.array([0, -1])
RECHTS = np.array([0, 1])
UNTEN_LINKS = np.array([1, -1])
UNTEN = np.array([1, 0])
UNTEN_RECHTS = np.array([1, 1])

RICHTUNGEN = [OBEN_LINKS, OBEN, OBEN_RECHTS, LINKS, RECHTS, UNTEN_LINKS, UNTEN, UNTEN_RECHTS]

class Stellung(np.ndarray):

    def __new__(cls):
        stellung = np.ndarray.__new__(cls, (8,8), dtype=np.int8)
        stellung.fill(0)
        return stellung
      
    def __array_wrap__(self, array, context=None, return_scalar=False):
        return array[()]
      
    def schwarz(self, liste):
        for index in liste:
            self[index[0],index[1]] = -1

    def weiss(self, liste):
        for index in liste:
            self[index[0],index[1]] = 1  

    def grundstellung(self):
        self.weiss([(3,3),(4,4)])
        seld.schwarz([(3,4),(4,3)])

    def moegliche_zuege(self, farbe):
        """
        Finds all possible moves for the given color.
        """
        zuege = []
        for z in range(8):
            for s in range(8):
                if self[z, s] == 0:
                    r_liste = []
                    for richtung in RICHTUNGEN:
                        a, b = np.array([z, s]) + richtung
                        if 0 <= a < 8 and 0 <= b < 8 and self[a, b] == -1 * farbe:
                            if self.__einschluss(a, b, richtung, farbe):
                                r_liste.append(richtung)
                    if r_liste:
                        zuege.append(((z, s), r_liste))
        return zuege

    def __einschluss(self, z, s, richtung, farbe):
        """
        Checks if a move is valid by checking for flanking pieces of opposite color.
        """
        if not (0 <= z < 8 and 0 <= s < 8) or self[z, s] != -1 * farbe:
            assert False
        a, b = z, s
        while True:
            c, d = np.array([a, b]) + richtung
            if not (0 <= c < 8 and 0 <= d < 8):
                return False
            if self[c, d] == farbe:
                return True
            if self[c, d] == 0:
                return False
            a, b = c, d

    def zug_spielen(self, zug, farbe):
        """
        Applies a move to the board, flipping captured pieces.
        """
        z, s = zug[0]
        assert self[z, s] == 0
        neue_stellung = stellung.copy()
        self[z, s] = farbe
        for richtung in zug[1]:
            umzudrehende_steine = self.__eingeschlossene_steine(z, s, richtung, farbe)
            for stein in umzudrehende_steine:
                self[stein[0], stein[1]] = farbe
       
    def __eingeschlossene_steine(self, z, s, richtung, farbe):
        """
        Finds all pieces that would be flipped by a move.
        """
        a, b = np.array([z, s]) + richtung
        assert 0 <= a < 8 and 0 <= b < 8
        steine = [(a, b)]
        while True:
            c, d = np.array([a, b]) + richtung
            if not (0 <= c < 8 and 0 <= d < 8):
                assert False
            if self[c, d] == farbe:
                return steine
            if self[c, d] == 0:
                assert False
            steine.append((c, d))
            a, b = c, d
