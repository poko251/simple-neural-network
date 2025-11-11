import numpy as np 
import pandas as pd
import sys

sys.stdout = open("wynik.txt", "w", encoding="utf-8")

beta = 1
epoki = 1000
eta = 0.1


print(f"Ilość epok: {epoki}")
def sigmoid(s, beta):
    sigmoid = 1/(1 + np.exp(-s*beta))
    return sigmoid

def sigmoid_pochodna(s, beta):
    f = sigmoid(s, beta)
    return beta * f * (1-f)

def epsilon(d, y):
    epsilon = d - y 
    return epsilon

def e(d, y):
    eps = epsilon(d, y)
    return 0.5 *np.sum(eps**2)

#warstwa wyjsciowa

#pochodna waga

def pochodna_waga_wyjscie(d, y, beta, s, v):
    return -1 * epsilon(d, y) * sigmoid_pochodna(s, beta) * v

#pochodna bias

def pochodna_bias_wyjscie(d, y, beta, s):
    return -1 * epsilon(d, y) * sigmoid_pochodna(s, beta) * 1

#warstwa ukryta


def pochodna_waga_ukryta(d, y, beta, s_output, v, w, x):
    """
    d        - wektor wartości wzorcowych (d)
    y        - wektor aktualnych wyjść (y)
    beta     - parametr funkcji sigmoidalnej
    s_output - wektor sum ważonych w neuronach wyjściowych (s_h_plus_m)
    v        - wyjście neuronu ukrytego (v_i)
    w        - wektor wag między tym neuronem ukrytym a neuronami wyjściowymi (w_h+m^(i))
    x        - wejście do neuronu ukrytego (x_j)
    """

    delta_output = -epsilon(d, y) * sigmoid_pochodna(s_output, beta) * w

    suma = np.sum(delta_output)

    delta_hidden = suma * beta * v * (1 - v)

    return delta_hidden * x


def pochodna_bias_ukryta(d, y, beta, s_output, v, w):

    delta_output = -epsilon(d, y) * sigmoid_pochodna(s_output, beta) * w

    suma = np.sum(delta_output)

    delta_hidden = suma * beta * v * (1 - v)

    return delta_hidden


def wagi():
    # Wagi warstwy ukrytej
    W1_1 = np.random.rand()  # do neuronu v1 z x1
    W1_2 = np.random.rand()  # do neuronu v1 z x2
    W2_1 = np.random.rand()  # do neuronu v2 z x1
    W2_2 = np.random.rand()  # do neuronu v2 z x2

    # Biasy warstwy ukrytej
    B1 = np.random.rand()
    B2 = np.random.rand()

    # Wagi warstwy wyjściowej
    W3_1 = np.random.rand()  # do neuronu y1 z v1
    W3_2 = np.random.rand()  # do neuronu y1 z v2
    W4_1 = np.random.rand()  # do neuronu y2 z v1
    W4_2 = np.random.rand()  # do neuronu y2 z v2

    # Biasy warstwy wyjściowej
    B3 = np.random.rand()
    B4 = np.random.rand()

    return W1_1, W1_2, W2_1, W2_2, B1, B2, W3_1, W3_2, W4_1, W4_2, B3, B4



def przod(W1_1, W1_2, W2_1, W2_2, B1, B2,
          W3_1, W3_2, W4_1, W4_2, B3, B4, x):

    # x = [x1, x2, d1, d2]
    x1, x2, d1, d2 = x

    # warstwa ukryta 
    sv_1 = W1_1 * x1 + W1_2 * x2 + B1
    sv_2 = W2_1 * x1 + W2_2 * x2 + B2
    v1 = sigmoid(sv_1, beta)
    v2 = sigmoid(sv_2, beta)

    #  warstwa wyjściowa 
    sy_1 = W3_1 * v1 + W3_2 * v2 + B3
    sy_2 = W4_1 * v1 + W4_2 * v2 + B4
    y1 = sigmoid(sy_1, beta)
    y2 = sigmoid(sy_2, beta)

    #  błędy 
    e1 = epsilon(d1, y1)
    e2 = epsilon(d2, y2)

    return sv_1, sv_2, v1, v2, sy_1, sy_2, y1, y2, e1, e2


def wstecz(W1_1, W1_2, W2_1, W2_2, B1, B2,
           W3_1, W3_2, W4_1, W4_2, B3, B4,
           x, sv_1, sv_2, v1, v2, sy_1, sy_2, y1, y2, eta):

    # Dane wejściowe
    x1, x2, d1, d2 = x
    d = np.array([d1, d2])
    y = np.array([y1, y2])

    # Pochodne dla warstwy wyjściowej
    gW3_1 = pochodna_waga_wyjscie(d1, y1, beta, sy_1, v1)
    gW3_2 = pochodna_waga_wyjscie(d1, y1, beta, sy_1, v2)
    gB3   = pochodna_bias_wyjscie(d1, y1, beta, sy_1)

    gW4_1 = pochodna_waga_wyjscie(d2, y2, beta, sy_2, v1)
    gW4_2 = pochodna_waga_wyjscie(d2, y2, beta, sy_2, v2)
    gB4   = pochodna_bias_wyjscie(d2, y2, beta, sy_2)

    # Pochodne dla warstwy ukrytej 
    gW1_1 = pochodna_waga_ukryta(d, y, beta, np.array([sy_1, sy_2]), v1, np.array([W3_1, W4_1]), x1)
    gW1_2 = pochodna_waga_ukryta(d, y, beta, np.array([sy_1, sy_2]), v1, np.array([W3_1, W4_1]), x2)
    gB1   = pochodna_bias_ukryta(d, y, beta, np.array([sy_1, sy_2]), v1, np.array([W3_1, W4_1]))

    gW2_1 = pochodna_waga_ukryta(d, y, beta, np.array([sy_1, sy_2]), v2, np.array([W3_2, W4_2]), x1)
    gW2_2 = pochodna_waga_ukryta(d, y, beta, np.array([sy_1, sy_2]), v2, np.array([W3_2, W4_2]), x2)
    gB2   = pochodna_bias_ukryta(d, y, beta, np.array([sy_1, sy_2]), v2, np.array([W3_2, W4_2]))

    # Aktualizacja wag 
    W3_1 -= eta * gW3_1
    W3_2 -= eta * gW3_2
    B3   -= eta * gB3

    W4_1 -= eta * gW4_1
    W4_2 -= eta * gW4_2
    B4   -= eta * gB4

    W1_1 -= eta * gW1_1
    W1_2 -= eta * gW1_2
    B1   -= eta * gB1

    W2_1 -= eta * gW2_1
    W2_2 -= eta * gW2_2
    B2   -= eta * gB2

    return (W1_1, W1_2, W2_1, W2_2, B1, B2,
            W3_1, W3_2, W4_1, W4_2, B3, B4)




dane = pd.read_csv("zbior_treningowy.csv")
X = dane[['Height', 'Weight']].values
D = dane[['D1', 'D2']].values

# Normalizacja 
X[:,0] = X[:,0] / np.max(X[:,0])  # Height
X[:,1] = X[:,1] / np.max(X[:,1])  # Weight

W = wagi()




for ep in range(epoki):
    blad = 0.0
    for i in range(len(X)):
        x1, x2 = X[i]
        d1, d2 = D[i]
        x = [x1, x2, d1, d2]

        # Przód
        sv_1, sv_2, v1, v2, sy_1, sy_2, y1, y2, e1, e2 = przod(*W, x)

        # Wstecz i aktualizacja
        W = wstecz(*W, x, sv_1, sv_2, v1, v2, sy_1, sy_2, y1, y2, eta)

        # Blad
        blad += e(np.array([d1, d2]), np.array([y1, y2]))

    if ep % 100 == 0:
        print(f"Epoka {ep} | Średni błąd = {blad / len(X)}")
print(f"Epoka {ep} | Średni błąd = {blad / len(X)}")

# Wagi po treningu
print("\n KOŃCOWE WAGI I BIASY PO TRENINGU")
print(f"W1_1={W[0]:.6f}, W1_2={W[1]:.6f}, W2_1={W[2]:.6f}, W2_2={W[3]:.6f}")
print(f"B1={W[4]:.6f}, B2={W[5]:.6f}")
print(f"W3_1={W[6]:.6f}, W3_2={W[7]:.6f}, W4_1={W[8]:.6f}, W4_2={W[9]:.6f}")
print(f"B3={W[10]:.6f}, B4={W[11]:.6f}")



#dane testwe

dane_testowe = pd.read_csv("zbior_testowy.csv")
X_test = dane_testowe[['Height', 'Weight']].values
D_test = dane_testowe[['D1', 'D2']].values

# Normalizacja 
X_test[:,0] = X_test[:,0] / np.max(X_test[:,0])  # Height
X_test[:,1] = X_test[:,1] / np.max(X_test[:,1])  # Weight


print("\nTEST PO TRENINGU")
for i in range(len(X_test)):
    x1, x2 = X_test[i]
    d1, d2 = D_test[i]
    x = [x1, x2, d1, d2]
    _, _, _, _, _, _, y1, y2, _, _ = przod(*W, x)
    print(f"Wejście: {x1:.1f}, {x2:.1f} -> Wyjście: [{y1:.3f}, {y2:.3f}] | Oczekiwane: [{d1}, {d2}]")


