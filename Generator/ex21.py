import scipy.io
import numpy as np
import matplotlib.pyplot as plt
mat = scipy.io.loadmat('data21.mat')

class generator:
    def __init__(self, learning_rate=0.003):
        self.A1 = mat['A_1'] # weights of the hidden layer, size 128x10
        self.B1 = mat['B_1'] # bias of the hidden layer, size 128x1

        self.A2 = mat['A_2'] # weights of the output layer, size 784x128
        self.B2 = mat['B_2'] # bias of the output layer, size 784x1

    def generate(self, Z):
        self.Z = Z # input noise, size 10x1
        self.feed_forward()

        # self.X = self.X.reshape(28,28).T # output image, size 28x28

    def feed_forward(self):
        self.W1 = self.A1 @ self.Z + self.B1
        self.Z1 = np.maximum(0, self.W1) # ReLU activation function
        self.W2 = self.A2 @ self.Z1 + self.B2

        self.X = 1/(1+np.exp(self.W2)) # sigmoid activation function


if __name__ == '__main__':
    # Generate random realizations
    N = 100  # Number of realizations
    random_realizations = np.random.randn(N, 10)

    # Subplot configuration
    num_rows = int(np.ceil(N / 10))
    num_cols = min(N, 10)

    # Generate and plot images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    axes = axes.ravel()

    g = generator()
    for i in range(N):
        g.generate(random_realizations[i].reshape(10, 1))
        axes[i].imshow(g.X.reshape(28,28).T, cmap='gray')
        # axes[i].axis('off')

    plt.tight_layout()
    plt.show()