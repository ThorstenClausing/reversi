import torch
from torch.utils.data import DataLoader
from torch import nn
import pickle
import numpy as np
from bewertungsnetz import Bewertungsdaten, Bewertungsnetz

training_liste = []
test_liste = []
zaehler = 0
bewertungen = {}

dateiliste = ['../reversi' + str(i) + '.of' for i in range(128)]
for datei in datei_liste:
    with (open(datei,'rb')) as f:
        bewertungen = pickle.load(f)
    for stellung_bytes in bewertungen.keys():
        stellung = np.frombuffer(stellung_bytes, dtype=np.int8)
        bewertung = bewertungen[stellung_bytes][0]/bewertungen[stellung_bytes][1]
        if not zaehler % 2:
            trainig_liste.append((torch.from_numpy(stellung), torch.tensor(bewertung)))
        else:
            test_liste.append((torch.from_numpy(stellung), torch.tensor(bewertung)))
        zaehler += 1


training_daten = Bewertungsdaten(training_liste)
test_daten = Bewertungsdaten(test_liste)
training_datengeber = DataLoader(training_daten, batch_size=1024, shuffle=True)
test_datengeber = DataLoader(test_daten, batch_size=1024, shuffle=True)

modell = Bewertungsnetz()

def train_loop(datengeber, modell, verlustfunktion, optimizer):
    size = len(datengeber.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    modell.train()
    for batch, (X, y) in enumerate(datengeber):
        # Compute prediction and loss
        vorhersage = modell(X)
        verlust = verlustfunktion(vorhersage, y)

        # Backpropagation
        verlust.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 1000 == 0:
            verlust, current = verlust.item(), batch * 1024 + len(X)
            print(f"loss: {verlust:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(datengeber, modell, verlustfunktion):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    modell.eval()
    size = len(datengeber.dataset)
    num_batches = len(datengeber)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in datengeber:
            vorhersage = modell(X)
            test_loss += verlustfunktion(vorhersage, y).item()
            
    test_loss /= num_batches
    # r_quadrat implementieren!
    print(f"Test Error: \n Durchschnittsverlust: {(test_loss):>8f}%, R_quadrat: {r_quadrat:>0.3f} \n")

optimierer = torch.optim.SGD(modell.parameters(), lr=0.001)
epochen = 4
for t in range(epochen):
    print(f"Epoche {t+1}\n-------------------------------")
    train_loop(training_datengeber, modell, nn.MSELoss(), optimierer)
    test_loop(test_datengeber, modell, nn.MSELoss())

torch.save(optimierer.state_dict(), 'gewichte')
print("Ende")

