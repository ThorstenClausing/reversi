from spiellogik_alternativ import moegliche_zuege as mz_neu
from spiellogik_alternativ import zug_spielen as zs_neu
from spiellogik import moegliche_zuege as mz_alt
from spiellogik import zug_spielen as zs_alt
from spiellogik import SCHWARZ
import numpy as np

stellung = np.zeros([8,8])
stellung[2,4] = stellung[3,3] = stellung[3,4] = stellung[4,4] = 1
stellung[4,3] = -1

def test_func():
  for i in range(len(mz_neu(stellung))):
    assert mz_neu(stellung)[i][0] == mz_alt(stellung)[i][0]
    for j in range(len(mz_neu(stellung)[i][1])):
      assert (mz_neu(stellung)[i][1][j] == mz_alt(stellung)[i][1][j]).all()
  assert (zs_neu(stellung, mz_neu(stellung)[0],SCHWARZ) == zs_alt(stllung, mz_alt(stellung)[0],SCHWARZ)).all()
