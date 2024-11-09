import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.colors as colors


H = 0.03
lam = 0.3
stars = loadmat('data32.mat')['stars']
circles = loadmat('data32.mat')['circles']

def K(X,Y,h=H):
    return np.exp(-np.square(np.linalg.norm(X-Y))/h)

s = np.block([[stars],
              [circles]])

A = np.zeros((s.shape[0],s.shape[0]))
for k in range(s.shape[0]):
    for l in range(s.shape[0]):
        A[l,k] = K(s[k],s[l])

B = np.zeros((s.shape[0],1))
for k in range(stars.shape[0]):
    B[k] = 1
for k in range(circles.shape[0]):
    B[k+stars.shape[0]] = -1

coeffs = np.linalg.solve(A+lam*np.eye(42),B)



def f_hat(X):
    f_h = 0

    for k, ai in enumerate(coeffs):
        f_h += np.dot(ai,K(X,s[k]))

    return f_h


# Define the range for x and y values
x_values = np.linspace(-1.5, 1.5, 500)
y_values = np.linspace(-0.5, 1.5, 500)

# Create an empty 2D array to store the function outputs
Z = np.zeros((len(x_values), len(y_values)))

# Evaluate the function for each point in the 2D space
for i, y in enumerate(y_values):
    for j, x in enumerate(x_values):
        fh = f_hat(np.array([x, y]))

        Z[i, j] = np.where(fh > 0, 1, -1)

# Create a color map to show the classification colors of each grid point
colors_with_alpha = [(1, 1, 0, 0.7), (0, 0, 1, 0.7)]  # Yellow and blue with transparency
cmap = colors.ListedColormap(colors_with_alpha)

# Create the plot
plt.figure(figsize=(8, 6))
plt.scatter(circles[:,0],circles[:,1],c='y')
plt.scatter(stars[:,0],stars[:,1],c='b', marker='*')
plt.imshow(Z, extent=[x_values.min(), x_values.max(), y_values.min(), y_values.max()], origin='lower', cmap=cmap)

# Add colorbar and labels
plt.colorbar(ticks=[-1, 1], label='φ̂(X)')
plt.xlabel('x₁')
plt.ylabel('x₂')

# Show the plot
plt.show()
