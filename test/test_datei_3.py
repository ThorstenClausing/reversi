from spiellogik_alternativ import moegliche_zuege, zug_spielen as mz_neu, zs_neu
from spiellogik import moegliche_zuege, zug_spielen as mz_alt, zs_alt
import numpy as np

stellung = np.zeros([8,8])
stellung[2,4] = stellung[3,3] = stellung[3,4] = stellung[4,4] = 1
stellung[4,3] = -1

def test_func():
  assert mz_neu(stellung) == mz_alt(stellung)
  assert zs_neu(stellung, mz_neu(stellung)[0],SCHWARZ) == zs_alt(stllung, mz_alt(stellung)[0],SCHWARZ)
