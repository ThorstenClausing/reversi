import os
import sys

# directory reach
directory = os.path(__file__).abspath()

# setting path
sys.path.append(directory.parent.parent)


from spiellogik import GRUNDSTELLUNG, moegliche_zuege

def test_func():
  zuege = []
  for zug in moegliche_zuege(GRUNDSTELLUNG):
    zuege.append(zug[0])
  assert (2,4) in zuege
  assert (3,5) in zuege
  assert (4,2) in zuege
  assert (5,3) in zuege
