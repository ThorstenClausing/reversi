from spiellogik_alternativ import moegliche_zuege as mz_neu
from spiellogik_alternativ import zug_spielen as zs_neu
from spiellogik import moegliche_zuege as mz_alt
from spiellogik import zug_spielen as zs_alt
from spiellogik import SCHWARZ
import numpy as np

stellung = np.zeros([8,8])
stellung[3,3] = stellung[4,2] = stellung[4,4] = stellung[5,2] = stellung[6,2] = 1
stellung[3,4] = stelung[4,3] = -1

def test_func():
  assert len(mz_neu(stellung)) == len(mz_alt(stellung))
  for i in range(len(mz_neu(stellung))):
    assert mz_neu(stellung)[i][0] == mz_alt(stellung)[i][0]
    assert len(mz_neu(stellung)[i][1]) == len(mz_alt(stellung)[i][1])
    assert (zs_neu(stellung, mz_neu(stellung)[i],SCHWARZ) == zs_alt(stellung, mz_alt(stellung)[i],SCHWARZ)).all()
    for j in range(len(mz_neu(stellung)[i][1])):
      assert (mz_neu(stellung)[i][1][j] == mz_alt(stellung)[i][1][j]).all()
