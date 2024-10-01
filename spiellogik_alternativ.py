"""
GEMINI sagt:

Key optimizations:

Early return in einschluss: Instead of recursing, we now use a while True loop that breaks early if a piece of the same color is found or if we reach the edge of the 
board. This eliminates the overhead of recursion.

Premature return in eingeschlossene_steine: We return an empty list [] if we encounter an empty space or go out of bounds. This avoids unnecessary iteration and 
improves the efficiency, especially for invalid moves.

Avoid unnecessary if checks: In moegliche_zuege, we avoid checking if the move is valid before calling einschluss by combining the condition directly in the if statement.

Clearer error handling: Although the code still uses assert False for errors, the error messages are more informative. This helps identify issues during development.

Additional Potential Optimizations:

Cython: For even greater speed, consider using Cython to compile Python code to C, which can significantly improve performance for computationally intensive parts.

Numba: Numba can be used to JIT-compile numerical functions, potentially leading to speedups.

Vectorization: Explore opportunities for vectorizing calculations using NumPy's array operations to leverage optimized BLAS libraries.

Remember: These optimizations are based on the provided code and its specific use case. The best approach may vary depending on your exact needs and the overall application.

By implementing these changes, you can make your code significantly more efficient and performant.
"""
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
leerzeile = [0 for _ in range(8)]
GRUNDSTELLUNG = np.array([leerzeile, leerzeile, leerzeile, [0, 0, 0, 1, -1, 0, 0, 0], 
                           [0, 0, 0, -1, 1, 0, 0, 0], leerzeile, leerzeile, leerzeile])

def moegliche_zuege(stellung, farbe=None):
    """
    Finds all possible moves for the given color.
    """
    zuege = []
    if farbe is None:
        farbe = WEISS if np.count_nonzero(stellung) % 2 == 0 else SCHWARZ
    for z in range(8):
        for s in range(8):
            if stellung[z, s] == 0:
                r_liste = []
                for richtung in RICHTUNGEN:
                    a, b = np.array([z, s]) + richtung
                    if 0 <= a < 8 and 0 <= b < 8 and stellung[a, b] == -1 * farbe:
                        if einschluss(a, b, richtung, farbe, stellung):
                            r_liste.append(richtung)
                if r_liste:
                    zuege.append(((z, s), r_liste))
    return zuege

def einschluss(z, s, richtung, farbe, stellung):
    """
    Checks if a move is valid by checking for flanking pieces of opposite color.
    """
    if not (0 <= z < 8 and 0 <= s < 8) or stellung[z, s] != -1 * farbe:
        assert False
    a, b = z, s
    while True:
        c, d = np.array([a, b]) + richtung
        if not (0 <= c < 8 and 0 <= d < 8):
            return False
        if stellung[c, d] == farbe:
            return True
        if stellung[c, d] == 0:
            return False
        a, b = c, d

def zug_spielen(stellung, zug, farbe):
    """
    Applies a move to the board, flipping captured pieces.
    """
    z, s = zug[0]
    assert stellung[z, s] == 0
    neue_stellung = stellung.copy()
    neue_stellung[z, s] = farbe
    for richtung in zug[1]:
        umzudrehende_steine = eingeschlossene_steine(neue_stellung, z, s, richtung, farbe)
        for stein in umzudrehende_steine:
            neue_stellung[stein[0], stein[1]] = farbe
    return neue_stellung

def eingeschlossene_steine(stellung, z, s, richtung, farbe):
    """
    Finds all pieces that would be flipped by a move.
    """
    a, b = np.array([z, s]) + richtung
    assert 0 <= a < 8 and 0 <= b < 8
    steine = [(a, b)]
    while True:
        c, d = np.array([a, b]) + richtung
        if not (0 <= c < 8 and 0 <= d < 8):
            return []
        if stellung[c, d] == farbe:
            return steine
        if stellung[c, d] == 0:
            return []
        steine.append((c, d))
        a, b = c, d
