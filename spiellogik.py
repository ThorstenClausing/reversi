import numpy as np
from numba import jit
#from struct import pack

AM_ZUG = 1
NICHT_AM_ZUG = -1
LEER = 0

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

    def __new__(cls):
        stellung = np.ndarray.__new__(cls, (8,8), dtype=np.int8)
        stellung.fill(LEER)
        return stellung
      
    def __array_wrap__(self, array, context=None, return_scalar=False):
        return array[()]
      
    def __nicht_am_zug(self, liste):
        for index in liste:
            self[index[0],index[1]] = NICHT_AM_ZUG

    def __am_zug(self, liste):
        for index in liste:
            self[index[0],index[1]] = AM_ZUG  

    def grundstellung(self):
        self.__am_zug([(3,3),(4,4)])
        self.__nicht_am_zug([(3,4),(4,3)])

    def moegliche_zuege(self):
        """
        Finds all possible moves for the player who is to move.
        """
        zuege = []
        for z in range(8):
            for s in range(8):
                if self[z, s] == LEER:
                    r_liste = []
                    for richtung in RICHTUNGEN:
                        a, b = np.array([z, s]) + richtung
                        if 0 <= a < 8 and 0 <= b < 8 and self[a, b] == NICHT_AM_ZUG:
                            if self.__einschluss(a, b, richtung):
                                r_liste.append(richtung)
                    if r_liste:
                        zuege.append(((z, s), r_liste))
        return zuege

    @jit
    def __einschluss(self, z, s, richtung):
        """
        Checks if a move is valid by checking for flanking pieces of opposite color.
        """
        if not (0 <= z < 8 and 0 <= s < 8) or self[z, s] != NICHT_AM_ZUG:
            assert False
        a, b = z, s
        while True:
            c, d = np.array([a, b]) + richtung
            if not (0 <= c < 8 and 0 <= d < 8):
                return False
            if self[c, d] == AM_ZUG:
                return True
            if self[c, d] == LEER:
                return False
            a, b = c, d

    def zug_spielen(self, zug):
        """
        Applies a move to the board, flipping captured pieces.
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
  
    @jit             
    def __eingeschlossene_steine(self, z, s, richtung):
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
            if self[c, d] == AM_ZUG:
                return steine
            if self[c, d] == LEER:
                assert False
            steine.append((c, d))
            a, b = c, d

    def __gegenspieler_kommt_zum_zug(self):
        """
        Vertauscht AM_ZUG und NICHT_AM_ZUG.
        """
        with np.nditer(self, op_flags=['readwrite'], flags=['external_loop']) as it:
            for x in it:
                x[...] = -x
        
    def stellung_anzeigen(self):
        spalte = 0
        for feld in np.nditer(self):
            match(feld):
                case 1: print(' X', end='')
                case -1: print(' O', end='')
                case _: print(' -', end='')
            spalte += 1
            if spalte == 8:
                spalte = 0
                print('\n', end='')
    
   
    # def tobytes(self):      
    #     liste = []
    #     for idx in range(8):
    #         zahl = 0
    #         for b in np.nditer(self[idx,:]):
    #             zahl <<= 2
    #             if b == -1:
    #                 zahl += 3
    #             else:
    #                 zahl += b
    #         liste.append(zahl)
    #     return pack('>HHHHHHHH', liste[0],liste[1],liste[2],liste[3],liste[4],liste[5],liste[6],liste[7])
    

def als_kanonische_stellung(stellung):
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
    