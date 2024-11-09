from mnist import MNIST
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

mndata = MNIST('samples')
images, labels = mndata.load_training()

# index = random.randrange(0, len(images))  # choose an index ;-)
# print(mndata.display(images[index]))

images = np.array(images)/255
labels = np.array(labels)

index_0 = np.where(labels == 0)[0]
index_8 = np.where(labels == 8)[0]

eights = images[index_8]
zeros = images[index_0][:len(index_8)]

print(eights.shape)
print(zeros.shape)


class NN:
    def __init__(self, learning_rate=0.005):
        self.W1 = np.random.normal(0, 1/1084, (300, 784)) # weights of the hidden layer, initialized randomly
        self.b1 = np.zeros((300,1)) # biases of the hidden layer, initialized to zero

        self.W2 = np.random.normal(0, 1/301, (1, 300)) # weights of the output layer, initialized randomly
        self.b2 = np.zeros((1,1)) # bias of the output layer, initialized to zero

        self.learning_rate = learning_rate # learning rate

    def train(self, X, Y, question=1):
        self.X = X
        self.Y = Y
        self.feed_forward()
        if question == 1:
            self.y_pred = 1/(1+np.exp(-self.z2)) # sigmoid activation function
        elif question == 2:
            self.y_pred = max(0, self.z2)
        else:
            self.y_pred = self.z2

        self.backpropagation(question=question)
        return self.y_pred
    
    def test(self, X, Y, question=1):
        self.X = X
        self.Y = Y
        self.feed_forward()
        if question == 1:
            return 1/(1+np.exp(-self.z2)) # sigmoid activation function
        elif question == 2:
            return max(0, self.z2)
        else:
            return self.z2

    def feed_forward(self):
        self.z1 = self.W1 @ self.X + self.b1
        self.A1 = np.maximum(0, self.z1) # ReLU activation function
        self.z2 = self.W2 @ self.A1 + self.b2
          

    def backpropagation(self, question=1):
        if question == 1:
            if self.Y == 0:
                self.L = -np.log(1-self.y_pred)

                dz2 = self.y_pred.item()
            else:
                self.L = -np.log(self.y_pred)

                dz2 = self.y_pred.item()-1
        elif question == 2:
            if self.Y == 0:
                self.L = 0.5*self.y_pred**2

                dz2 = self.y_pred.item() * (self.z2 > 0)
            else:
                self.L = -self.y_pred

                dz2 = (self.z2 > 0)*1
        else:
            if self.Y == 0:
                self.L = np.exp(0.5*self.y_pred)

                dz2 = 0.5*self.L.item()
            else:
                self.L = np.exp(-0.5*self.y_pred)

                dz2 = -0.5*self.L.item()

            
        dW2 = (dz2 * self.A1).T
        db2 = np.array([dz2])
        db1 = dz2 * self.W2.T * (self.z1 > 0)
        dW1 = db1 @ self.X.T

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

# ==================== Initialization ====================
max_iter = 50 # maximum number of epochs

nn = NN() # create a neural network
cross_entropy_costs = [] # to store the cost per epoch
e0 = 0
e1 = 0

print('Cross Entropy training:')

# train the neural network
for epoch in tqdm(range(max_iter)):
    cost_epoch = 0
    np.random.shuffle(eights)
    np.random.shuffle(zeros)
    for i in range(len(index_8)):
        
        x0 = eights[i].reshape(-1,1)
        x1 = zeros[i].reshape(-1,1)

        u0 = nn.train(x0, 0)
        c1 = nn.L
        u1 = nn.train(x1, 1)
        c2 = nn.L

        cost_epoch += c1+c2
    cross_entropy_costs.append(cost_epoch/200)

flattened_costs = np.array(cross_entropy_costs).reshape(-1)
plt.plot(range(max_iter), flattened_costs)
plt.title('Cost per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()