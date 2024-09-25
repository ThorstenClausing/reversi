import numpy as np

WEISS = 1
SCHWARZ = -1

OBEN_LINKS = np.array([-1,-1])
OBEN = np.array([-1,0])
OBEN_RECHTS = np.array([-1,1])
LINKS = np.array([0,-1])
RECHTS = np.array([0,1])
UNTEN_LINKS = np.array([1,-1])
UNTEN = np.array([1,0])
UNTEN_RECHTS = np.array([1,1])

RICHTUNGEN = [OBEN_LINKS, OBEN, OBEN_RECHTS, LINKS, RECHTS, UNTEN_LINKS, UNTEN, UNTEN_RECHTS]
leerzeile = [0 for _ in range(8)]
GRUNDSTELLUNG = np.array([leerzeile, leerzeile,leerzeile,[0,0,0,1,-1,0,0,0],[0,0,0,-1,1,0,0,0],leerzeile,leerzeile,leerzeile])

def moegliche_zuege(stellung,farbe=None):
  zuege = []
  if farbe==None:
    if stellung.nonzero()[0].shape[0] % 2 == 0:
      farbe = WEISS
    else:
      farbe = SCHWARZ
  for z in range(8):
    for s in range(8):
      if stellung[z,s] == 0:
        r_liste = []
        for richtung in RICHTUNGEN:
          a,b = np.array([z,s]) + richtung
          if 0 <= a < 8 and 0 <= b < 8:
            if stellung[a,b] == -1*farbe:
              if einschluss(a,b,richtung,farbe,stellung):
                r_liste.append(richtung)
        if len(r_liste) > 0:
          zuege.append(((z,s), r_liste))
  return zuege

def einschluss(z,s,richtung,farbe,stellung):
  assert stellung[z,s] == -1*farbe
  a,b = np.array([z,s]) + richtung
  if not (0 <= a < 8) or not (0 <= b < 8):
    return False
  if stellung[a,b] == -1*farbe:
      c,d = np.array([a,b]) + richtung
      if not (0 <= c < 8) or not (0 <= d < 8):
        return False
      else:
        return einschluss(a,b,richtung,farbe,stellung)
  elif stellung[a,b] == farbe:
      return True
  else:
      return False
