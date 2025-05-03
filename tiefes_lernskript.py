# -*- coding: utf-8 -*-
"""
Created on Sat May  3 13:18:29 2025

@author: User
"""
import torchrl.data as td
import torch
from torch import nn
from spieler import Lernender_Spieler
from bewertungsnetz import Bewertungsnetz
from partieumgebung import Partieumgebung

replay_buffer = td.ReplayBuffer(storage=td.LazyMemmapStorage(1000000), batch_size=128)
netz = Bewertungsnetz(replay_buffer)
spieler_schwarz = Lernender_Spieler(netz)
spieler_weiss = Lernender_Spieler(netz)
umgebung = Partieumgebung(spieler_schwarz, spieler_weiss, netz)

def train_loop(datengeber, modell, verlustfunktion, optimizer):
    modell.train()      # Unnecessary in this situation but added for best practices
    for batch, md in enumerate(datengeber):
        # Compute prediction and loss
        vorhersage = modell(md.stellungen)
        verlust = verlustfunktion(vorhersage, md.bewertungen)
        print(verlust.item())
        # Backpropagation
        verlust.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch >= 999: 
            break

verlustfunktion = nn.MSELoss()
optimator = torch.optim.Adam(netz.parameters(), lr=0.001)

anzahl_partien = 10

for _ in range(anzahl_partien):
    umgebung.partie_starten()
train_loop(replay_buffer, netz, verlustfunktion, optimator)
    
# Netz mit Daten aus replay_buffer trainieren
#for md in replay_buffer.sample():
#    print(md.stellungen)
#    print(md.bewertungen)