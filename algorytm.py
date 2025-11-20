import numpy as np 
import pandas as pd
from tqdm import tqdm

betas = [1.5]
etas = [0.1]
epochs = [300]
start_weight = 0.05
normalizacja = 1


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

    return delta_hidden


def wagi():
    # Wagi warstwy ukrytej
    W1_1 = np.random.uniform(-start_weight, start_weight) # do neuronu v1 z x1
    W1_2 = np.random.uniform(-start_weight, start_weight) # do neuronu v1 z x2
    W2_1 = np.random.uniform(-start_weight, start_weight)  # do neuronu v2 z x1
    W2_2 = np.random.uniform(-start_weight, start_weight)  # do neuronu v2 z x2

    # Biasy warstwy ukrytej
    B1 = np.random.uniform(-start_weight, start_weight)
    B2 = np.random.uniform(-start_weight, start_weight)

    # Wagi warstwy wyjściowej
    W3_1 = np.random.uniform(-start_weight, start_weight) # do neuronu y1 z v1
    W3_2 = np.random.uniform(-start_weight, start_weight)  # do neuronu y1 z v2
    W4_1 = np.random.uniform(-start_weight, start_weight)  # do neuronu y2 z v1
    W4_2 = np.random.uniform(-start_weight, start_weight) # do neuronu y2 z v2

    # Biasy warstwy wyjściowej
    B3 = np.random.uniform(-start_weight, start_weight)
    B4 = np.random.uniform(-start_weight, start_weight)

    return W1_1, W1_2, W2_1, W2_2, B1, B2, W3_1, W3_2, W4_1, W4_2, B3, B4



def przod(W1_1, W1_2, W2_1, W2_2, B1, B2,
          W3_1, W3_2, W4_1, W4_2, B3, B4, x, beta):

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
           x, sv_1, sv_2, v1, v2, sy_1, sy_2, y1, y2, eta, beta):

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



def start(betas, etas, epochs, normalizacja=0):
    for beta in tqdm(betas, desc="Beta"):
        for eta in tqdm(etas, desc="Etas", leave=False):
            for epoch in tqdm(epochs, desc="Epoch", leave=False):
                

                filename = f"wynik_beta_{beta}_eta_{eta}_ep_{epoch}_normalizacja_{normalizacja}.txt"

                with open(filename, "w", encoding="utf-8") as f:

                    print(f"Ilość epok: {epoch},", file=f)
                    print(f"Beta: {beta}", file=f)
                    print(f"Eta: {eta}", file=f)
                    print(f"Normalizacja: {normalizacja}", file=f)
                    print(f"Wagi losowane z zakresu ({-start_weight}, {start_weight})", file=f)
                    dane = pd.read_csv("zbior_treningowy.csv")
                    X = dane[['Height', 'Weight']].values
                    D = dane[['D1', 'D2']].values

                    #dane testwe

                    dane_testowe = pd.read_csv("zbior_testowy.csv")
                    X_test = dane_testowe[['Height', 'Weight']].values
                    D_test = dane_testowe[['D1', 'D2']].values

                    # Z-score

                    if normalizacja:
                        H_mean, H_std = np.mean(X[:,0]), np.std(X[:,0])
                        W_mean, W_std = np.mean(X[:,1]), np.std(X[:,1])

                        X[:,0] = (X[:,0] - H_mean) / H_std
                        X[:,1] = (X[:,1] - W_mean) / W_std

                        X_test[:,0] = (X_test[:,0] - H_mean) / H_std
                        X_test[:,1] = (X_test[:,1] - W_mean) / W_std




                    W = wagi()


                    for ep in tqdm(range(epoch), desc="Training", leave=False):
                        blad = 0.0
                        for i in range(len(X)):
                            x1, x2 = X[i]
                            d1, d2 = D[i]
                            x = [x1, x2, d1, d2]

                            # Przód
                            sv_1, sv_2, v1, v2, sy_1, sy_2, y1, y2, e1, e2 = przod(*W, x, beta)

                            # Wstecz i aktualizacja
                            W = wstecz(*W, x, sv_1, sv_2, v1, v2, sy_1, sy_2, y1, y2, eta, beta)

                            # Blad
                            blad += e(np.array([d1, d2]), np.array([y1, y2]))

                        if ep % 100 == 0:
                            print(f"Epoka {ep} | Średni błąd = {blad / len(X)}", file=f)
                    print(f"Epoka {ep} | Średni błąd = {blad / len(X)}", file=f)

                    # Wagi po treningu
                    print("\n KOŃCOWE WAGI I BIASY PO TRENINGU", file=f)
                    print(f"W1_1={W[0]:.6f}, W1_2={W[1]:.6f}, W2_1={W[2]:.6f}, W2_2={W[3]:.6f}",file=f)
                    print(f"B1={W[4]:.6f}, B2={W[5]:.6f}", file=f)
                    print(f"W3_1={W[6]:.6f}, W3_2={W[7]:.6f}, W4_1={W[8]:.6f}, W4_2={W[9]:.6f}",file=f)
                    print(f"B3={W[10]:.6f}, B4={W[11]:.6f}",file=f)



                    print("\nTEST PO TRENINGU",file=f)


                    cm = np.zeros((2,2), dtype=int)

                    for i in range(len(X_test)):
                        x1, x2 = X_test[i]
                        d1, d2 = D_test[i]
                        x = [x1, x2, d1, d2]
                        _, _, _, _, _, _, y1, y2, _, _ = przod(*W, x, beta)

                        pred = np.argmax([y1, y2])
                        true = np.argmax([d1, d2])


                        cm[true, pred] += 1


                        print(f"Wejście: {x1:.3f}, {x2:.3f} -> Wyjście: [{y1:.3f}, {y2:.3f}] | Oczekiwane: [{d1}, {d2}]| {'OK' if pred==true else 'BŁĄD'}",file=f)

                                                
                    print("\nMACIERZ POMYŁEK (confusion matrix)", file=f)
                    print("           Pred 0   Pred 1", file=f)
                    print(f"True 0      {cm[0,0]:5d}   {cm[0,1]:5d}", file=f)
                    print(f"True 1      {cm[1,0]:5d}   {cm[1,1]:5d}", file=f)

                 
                    poprawne = cm[0,0] + cm[1,1]
                    bledne   = cm[0,1] + cm[1,0]

                    print("\nPODSUMOWANIE", file=f)
                    print(f"Poprawne: {poprawne}", file=f)
                    print(f"Błędne:   {bledne}", file=f)
                        

start(betas, etas, epochs, normalizacja)



# 1 Wpływ normalizacji na dokładność klasyfikacji, (chcemy dobrać parametry dla bez normalizacji zeby uzyskac efekt jak z normalzacją)
#2 Wplyw liczby neuronów ukrytych : 0, 2
#3 Macierz pomyłek