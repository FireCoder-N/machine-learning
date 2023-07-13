import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def k_means_clustering(vectors, k, max_iterations=100):
    # Randomly initialize cluster centroids
    centroids = vectors[np.random.choice(range(len(vectors)), size=k, replace=False)]

    for _ in range(max_iterations):
        # Assign vectors to the nearest centroid
        distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
        cluster_labels = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.array([vectors[cluster_labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return cluster_labels, centroids

vectors = loadmat('data33.mat')['X']
vectors = vectors.T

# Apply k-means clustering
cluster_labels, centroids = k_means_clustering(vectors, 2)

count0 = 0
count1 = 0
for i in range(len(vectors)//2):
    if cluster_labels[i] == 0:
        count0 += 1
    else:
        count1 += 1

if count0 > count1:
    color0 = 'red'
    color1 = 'blue'
else:
    color0 = 'blue'
    color1 = 'red'
            
error_rb = 0
error_br = 0
for i in range(len(vectors)):
    if cluster_labels[i] == 0:
        ccolor = color0
    else:
        ccolor = color1

    if i < 100:
        mcolor = 'red'
        if ccolor != mcolor:
            error_rb += 1
    else:
        mcolor = 'blue'
        if ccolor != mcolor:
            error_br += 1


    plt.scatter(vectors[i, 0], vectors[i, 1], color=mcolor)
    plt.gca().add_artist(plt.Circle((vectors[i, 0], vectors[i, 1]), 0.1, color=ccolor, fill=False))

print('Error 1 (red classified as blue): {}%'.format(2*(error_rb) / len(vectors) * 100))
print('Error 2 (blue classified as red): {}%'.format(2*(error_br) / len(vectors) * 100))
print('Total error: {}%'.format((error_br + error_rb) / len(vectors) * 100))

plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('K-Means Clustering')
plt.grid(True)
plt.show()