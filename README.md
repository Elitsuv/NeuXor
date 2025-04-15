# 🧠 NeuXor — The XOR Neural Network
A minimal neural network built entirely from scratch using NumPy — designed to learn the classic XOR logic function.

> [!NOTE] 
> This project is experimental and intended for learning purposes only.

## 📌 Project Goal
To **demystify how neural networks work** under the hood by building one from scratch —  
no libraries like TensorFlow or PyTorch — just pure matrix math using NumPy.

This project solves the **XOR problem**, which is not linearly separable (i.e., it can't be solved by a straight line).  
We use **1 hidden layer** with **2 neurons**, and **sigmoid** activations.

---

## 📚 Learning Objectives

- Understand forward and backward propagation
- Learn how weights and biases are updated
- See how activation functions help neural nets learn
- Build confidence to create deeper nets from scratch

---

## 🧠 Neural Network Architecture

---

## 🧪 XOR Dataset

| Input (X1, X2) | Output |
|----------------|--------|
| 0, 0           |   0    |
| 0, 1           |   1    |
| 1, 0           |   1    |
| 1, 1           |   0    |

---

## 🧾 Code Structure

| Section            | What It Does                                      |
|--------------------|---------------------------------------------------|
| `X`, `Y`           | Input/Output values for XOR                       |
| `sigmoid()`        | Activation function                               |
| `derivative()`     | Used in backprop to adjust weights                |
| `w1`, `w2`, `b1`, `b2` | Random weights and biases for layers         |
| Forward Pass       | Calculates output                                 |
| Backward Pass      | Calculates how wrong and how to fix it            |
| Weight Update      | Adjusts weights and biases to improve accuracy    |
| Output             | Final predictions after training                  |

---

## 🛠 Technologies

- **Python 3.10+**
- **NumPy**

---

## 📈 Sample Output
#### Epoch 0 | Loss: 0.3312
#### Epoch 1000 | Loss: 0.1247
#### Epoch 2000 | Loss: 0.0611
...
#### Final Output: [[0.02] [0.98] [0.97] [0.05]]

## 📂 Run Locally

```bash
git clone (https://github.com/Elitsuv/NeuXor.git)
cd .
python network.py
