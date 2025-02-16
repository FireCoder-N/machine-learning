import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm
import ex21

mat = scipy.io.loadmat('data22.mat')
Xi = mat['X_i']
Xn = mat['X_n']

fig, axes = plt.subplots(4, 5, figsize=(10, 10))
axes = axes.ravel()
N = 400

T = np.concatenate((np.eye(N), np.zeros((N, 784 - N))), axis=1)  # T is a Nx784 matrix

epochs = 500
learning_rate = 0.0005

for i in range(4):
    xi = Xi[:, i].reshape(784, 1)
    xn = Xn[:, i].reshape(784, 1)
    xn = T @ xn

    g = ex21.generator()
    g.generate(np.random.randn(10, 1))

    costs = []

    for j in tqdm(range(epochs)):
        x = g.X.reshape(784, 1)

        norm_s = (xn - T @ x).T @ (xn - T @ x)
        J = N * np.log(norm_s) + g.Z.T @ g.Z # cost function

        U2 = (-2/norm_s) * T.T @ (xn - T @ x)
        V2 = U2 * -np.exp(g.W2) * g.X**2
        U1 = g.A2.T @ V2
        V1 = U1 * (g.W1 > 0)
        U0 = g.A1.T @ V1

        dZ = N * U0 + 2 * g.Z
        Z = g.Z - learning_rate * dZ

        costs.append(J)

        g.generate(Z)


    # xn = 1/(1+np.exp(-xn)) 
    xk = np.maximum(xn, 0)
    xk = np.concatenate((xn, np.zeros((784 - N, 1))), axis=0)

    k = 5*i
    axes[k].imshow(xi.reshape(28,28).T, cmap='gray')
    axes[k+1].imshow(xk.reshape(28,28).T, cmap='gray')
    axes[k+2].plot(range(epochs), np.array(costs).reshape(-1))
    axes[k+3].imshow(g.X.reshape(28,28).T, cmap='gray')
    axes[k+4].imshow((xi - g.X).reshape(28,28).T, cmap='gray')

    print('error: ', np.linalg.norm(xi - g.X))

plt.tight_layout()
plt.show()