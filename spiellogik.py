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

def zug_spielen(stellung,zug,farbe):
  z,s = zug[0]
  assert stellung[z,s] == 0
  neue_stellung = stellung.copy()
  neue_stellung[z,s] = farbe
  umzudrehende_steine = []
  for richtung in zug[1]:
    umzudrehende_steine.extend(eingeschlossene_steine(neue_stellung,z,s,richtung,farbe))
  for stein in umzudrehende_steine:
    neue_stellung[stein[0],stein[1]] = farbe
  return neue_stellung

def eingeschlossene_steine(stellung,z,s,richtung,farbe):
  a,b = np.array([z,s]) + richtung
  assert a in range(8) and b in range(8)
  steine = [(a,b)]
  c,d = np.array([a,b]) + richtung
  if (0 <= c < 8) and (0 <= d < 8):
    if stellung[c,d] == -1*farbe:
      steine.extend(eingeschlossene_steine(stellung,a,b,richtung,farbe))
    elif stellung[c,d] == farbe:
      return steine
    else:
      print(f'Fehler 1\t c = {c}, d = {d}, farbe = {farbe}\n',stellung[c,d],'\t',stellung)
      #Fehlerbehebung zu ergänzen
      assert False
  else:
    print(f'Fehler 2\t a = {a}, b = {b}, richtung = {richtung}, c = {c}, d = {d}, farbe = {farbe}\n',stellung)
    #Fehlerbehebung zu ergänzen
    assert False
  return steine
