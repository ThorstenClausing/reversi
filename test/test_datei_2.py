from spiellogik import moegliche_zuege
import numpy as np

stellung = np.zeros([8,8])
stellung[2,4] = stellung[3,3] = stellung[3,4] = stellung[4,4] = 1
stellung[4,3] = -1

def test_func():
  zuege = []
  for zug in (mz := moegliche_zuege(stellung)):
    zuege.append(zug[0])
  assert (2,3) in zuege
  assert (2,5) in zuege
  assert (4,5) in zuege
  assert len(mz) == 3
