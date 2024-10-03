import sys
import os

# getting the name of the directory
# where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

# adding the parent directory to 
# the sys.path.
sys.path.append(parent)

from spiellogik import GRUNDSTELLUNG, moegliche_zuege

def test_func():
  zuege = []
  for zug in moegliche_zuege(GRUNDSTELLUNG):
    zuege.append(zug[0])
  assert (2,4) in zuege
  assert (3,5) in zuege
  assert (4,2) in zuege
  assert (5,3) in zuege
