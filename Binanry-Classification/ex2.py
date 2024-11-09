import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ex1 import Test

# ==================== Training data ====================
K = 200 # Number of samples

# following f₀ distribution
train_f0_samples_x1 = np.random.normal(0, 1, K) # Generate K samples from N(0,1) for x₁
train_f0_samples_x2 = np.random.normal(0, 1, K) # Generate K samples from N(0,1) for x₂
train_f0_samples = np.vstack((train_f0_samples_x1, train_f0_samples_x2)).T # Stack the samples together, so that each row is a sample X=[x₁, x₂]

# similarly, following f₁ distribution
train_f1_samples_x1 = np.random.choice([-1, 1], size=K) # Generate K samples from {−1,1} for x₁
train_f1_samples_x1 = np.random.normal(train_f1_samples_x1, 1) # then each sample x₁ takes its value from N(−1,1) or N(1,1), based on the previous step
#same for x₂
train_f1_samples_x2 = np.random.choice([-1, 1], size=K) 
train_f1_samples_x2 = np.random.normal(train_f1_samples_x2, 1)

train_f1_samples = np.vstack((train_f1_samples_x1, train_f1_samples_x2)).T # Stack the samples together

# ==================== Neural network ====================
class NN:
    def __init__(self, learning_rate=0.003):
        self.W1 = np.random.normal(0, 1/22, (20, 2)) # weights of the hidden layer, initialized randomly
        self.b1 = np.zeros((20,1)) # biases of the hidden layer, initialized to zero

        self.W2 = np.random.normal(0, 1/21, (1, 20)) # weights of the output layer, initialized randomly
        self.b2 = np.zeros((1,1)) # bias of the output layer, initialized to zero

        self.learning_rate = learning_rate # learning rate

    def train(self, X, Y, question=1):
        self.X = X
        self.Y = Y
        self.feed_forward()
        if question == 1:
            self.y_pred = 1/(1+np.exp(-self.z2)) # sigmoid activation function
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
max_iter = 500 # maximum number of epochs
test = Test() # test data. From question 1b

# ==================== Cross Entropy ====================
nn_cross_entropy = NN() # create a neural network
cross_entropy_costs = [] # to store the cost per epoch
cross_entropy_e0 = 0
cross_entropy_e1 = 0

print('Cross Entropy training:')

# train the neural network
for epoch in tqdm(range(max_iter)):
    cost_epoch = 0
    np.random.shuffle(train_f0_samples)
    np.random.shuffle(train_f1_samples)
    for i in range(200):
        
        x0 = train_f0_samples[i].reshape(2,1)
        x1 = train_f1_samples[i].reshape(2,1)

        u0 = nn_cross_entropy.train(x0, 0)
        c1 = nn_cross_entropy.L
        u1 = nn_cross_entropy.train(x1, 1)
        c2 = nn_cross_entropy.L

        cost_epoch += c1+c2
    cross_entropy_costs.append(cost_epoch/200)

flattened_costs = np.array(cross_entropy_costs).reshape(-1)
plt.plot(range(max_iter), flattened_costs)
plt.title('Cost per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()


# ==================== Exponential ====================
nn_exponential = NN() # create a neural network
exponential_costs = [] # to store the cost per epoch
exponential_e0 = 0
exponential_e1 = 0

print("Exponential training:")

# train the neural network
for epoch in tqdm(range(max_iter)):
    cost_epoch = 0
    np.random.shuffle(train_f0_samples)
    np.random.shuffle(train_f1_samples)
    for i in range(200):
        
        x0 = train_f0_samples[i].reshape(2,1)
        x1 = train_f1_samples[i].reshape(2,1)

        u0 = nn_exponential.train(x0, 0, 2)
        c1 = nn_exponential.L
        u1 = nn_exponential.train(x1, 1, 2)
        c2 = nn_exponential.L

        cost_epoch += c1+c2
    exponential_costs.append(cost_epoch/200)

flattened_costs = np.array(exponential_costs).reshape(-1)
plt.plot(range(max_iter), flattened_costs)
plt.title('Cost per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.show()

# ==================== Testing ====================
print("Bays test:")
test.Test() # optimal decision boundary (baysian method)
print()

# cross entropy testing
for i in range(test.K):
    x0 = test.f0_samples[i].reshape(2,1)
    u0 = nn_cross_entropy.test(x0, 0)
    if u0 > 0.5 or (u0 == 0.5 and np.random.rand() > 0.5):
        cross_entropy_e0 += 1

    x1 = test.f1_samples[i].reshape(2,1)
    u1 = nn_cross_entropy.test(x1, 1)
    if u1 < 0.5 or (u1 == 0.5 and np.random.rand() > 0.5):
        cross_entropy_e1 += 1

print("cross entropy:")
print("data under H0 misclassified as H1 (e0): ", cross_entropy_e0/test.K)
print("data under H1 misclassified as H0 (e1): ", cross_entropy_e1/test.K)
print("Total error (e):", 0.5*(cross_entropy_e0+cross_entropy_e1)/test.K)
print()

# exponential testing
for i in range(test.K):
    x0 = test.f0_samples[i].reshape(2,1)
    u0 = nn_exponential.test(x0, 0, 2)
    if u0 > 0.5 or (u0 == 0.5 and np.random.rand() > 0.5):
        exponential_e0 += 1

    x1 = test.f1_samples[i].reshape(2,1)
    u1 = nn_exponential.test(x1, 1, 2)
    if u1 < 0.5 or (u1 == 0.5 and np.random.rand() > 0.5):
        exponential_e1 += 1

print("exponential:")
print("data under H0 misclassified as H1 (e0): ", exponential_e0/test.K)
print("data under H1 misclassified as H0 (e1): ", exponential_e1/test.K)
print("Total error (e):", 0.5*(exponential_e0+exponential_e1)/test.K)
print()