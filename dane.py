import pandas as pd
from sklearn.model_selection import train_test_split

dane = pd.read_csv("dane.csv", nrows=100)

dane["Sex"] = dane["Sex"].map({"Male": 1, "Female": 0})

# poprawne tworzenie zmiennych D1 i D2
dane["D1"] = (dane["Sex"] == 0).astype(int)
dane["D2"] = (dane["Sex"] == 1).astype(int)

dane = dane[["Height", "Weight", "D1", "D2"]]

train, test = train_test_split(dane, test_size=0.3, random_state=42)

train.to_csv("zbior_treningowy.csv", index=False)
test.to_csv("zbior_testowy.csv", index=False)
