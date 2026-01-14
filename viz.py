import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("zbior_treningowy.csv")


plt.scatter(df["Height"], df["Weight"], c=df["D1"])
plt.xlabel("Wzrost")
plt.ylabel("Waga")
plt.title("Dane")
plt.show()
