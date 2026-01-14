# Multi-Layer Perceptron from Scratch - University Project

The goal was to implement a Multi-Layer Perceptron (MLP) neural network completely from scratch. 

The implementation strictly uses the **NumPy** library for mathematical operations and follows the specific notation and algorithms provided during course lectures. No high-level frameworks (like TensorFlow or PyTorch) were used.


### 1. Network Structure
* **Input Layer:** 2 neurons ($x_1, x_2$)  (Height, Weight).
* **Hidden Layer:** 2 neurons ($v_1, v_2$) with sigmoid activation.
* **Output Layer:** 2 neurons ($y_1, y_2$) representing classes.

### 2. Mathematical Foundation
* **Activation Function:** Sigmoid.
* **Error Function:** Mean Squared Error (MSE).
* **Weight Update:** Gradient Descent with learning rate. All gradients are derived manually using the chain rule according to lecture slides.

## Research & Analysis
The project focuses on three primary experimental objectives:

#### 1. Impact of Normalization
Analysis of the network's performance with and without Z-score normalization. The goal was to find specific $\beta$ and $\eta$ parameters that allow the model to converge on raw data as effectively as it does on normalized data.

#### 2. Hidden Layer Configuration (0 vs 2 Neurons)
Comparison of model complexity:
0 Hidden Neurons:Testing the model as a simple linear classifier.
2 Hidden Neurons: Standard MLP architecture for non-linear decision boundaries.


##  How to Run

### Prerequisites
* Python 3.x
* NumPy
* Pandas
* tqdm

2. Run the script:
   ```bash
   python main.py
