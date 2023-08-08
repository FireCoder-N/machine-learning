# K-Means Clustering

This repository contains an algorithm to perform K-means clustering. There is also an example, where a bunch of 2-dimanesinal data are seperated in 2 clusters.

## Introduction

The k-means algorithm is one of the most popular data clustering algorithms and
belongs to the category of unsupervised learning. By this terminology we mean that we do not carry out some training and the algorithm is self-improved. Often the "right answer" (i.e. in which a category a particular data point belongs) is unknown, or in cases where it is known, it is only used retrospectively as an evaluation, as is the case in the given example. The goal of the algorithm is to find k centroids, where k is one predetermined number and divide the data into an equal number of groups.

## Algorithm Description

The K-Means algorithm works as follows:

1. **Initialization**: (Randomly) choose k points from the dataset as the initial centroids.

2. **Assignment**: Assign each data point to the nearest centroid based on some distance metric, commonly the Euclidean distance. This step creates k clusters, with each data point belonging to the cluster represented by the nearest centroid.

3. **Update Centroids**: Recalculate the centroids of each cluster by taking the mean of all data points belonging to that cluster. This step moves the centroids to the center of their respective clusters.

4. **Iteration**: Repeat steps 2 and 3 until convergence is reached, i.e., until the centroids no longer change significantly, or a maximum number of iterations is reached.

It is worth noting that the k-means algorithm is sensitive to the initialization of the centroids, as it can arrive at different solutions if it starts from different initial centroids. For this reason, the algorithm is often executed multiple times with different initializations to find the optimal solution, a practice that was not followed in the present work.

## Mathematical Notation

Let's denote the dataset as `X`, containing `n` data points with `m` features each, represented as a matrix with dimensions n x m. The centroids are represented as a matrix `C` with dimensions K x m.

The K-Means algorithm can be summarized with the following mathematical notation:

1. **Initialization**:

    Randomly select `K` data points from `X` as the initial centroids.

2. **Assignment**:

    For each data point `x` in `X`, find the nearest centroid `c` based on the Euclidean distance:
   
   ![Euclidean Distance](https://latex.codecogs.com/svg.latex?\inline&space;\text{distance}(x,&space;c)&space;=&space;\sqrt{\sum_{i=1}^{m}(x_i&space;-&space;c_i)^2})
   
   Assign each data point to the nearest centroid's cluster.

3. **Update Centroids**:

    Recalculate the centroids `C` based on the mean of all data points in their respective clusters:
   
   ![Update Centroids](https://latex.codecogs.com/svg.latex?\inline&space;c_i&space;=&space;\frac{1}{n_i}&space;\sum_{x&space;\in&space;C_i}&space;x),
   where n<sub>i</sub> is the number of data points in cluster C<sub>i</sub>.

4. **Iteration**:

    Repeat steps 2 and 3 until convergence (minimal centroid movement) or a maximum number of iterations is reached.

## Execution & Results

As an example, there are given n=200 data points, each with dimension m=2. It is also given that the first 100 points correspond to one cluster (e.g. red) and the other 100 points belong to the second cluster (e.g. blue). As stated before, the algorithm doesn't take into account this knowledge. After the execusion, K-means algorithm splits the data in two clusters, however we can't know whether the points that the algorithm classifies as cluster one correspond to actual cluster 1 and thus, we calculate both possible scenarios and keep the one that minimizes hte error.

The result of the execution (file 3a.py) is summarized in the following picture.

![Data Clustering](https://github.com/FireCoder-N/machine-learning/blob/main//K-means-clustering/3a.png?raw=true)

In this picture, each data point has two colors:
- the circle disk corresponds to the correct cluster (red or blue), following the 'secret information' from above.
- the border of each point (a colored ring) corresponds to the cluster as specified by the algorithm.

We can then count the errors performed:
```
error 1 (red points classified as blue): 28%
error 2 (blue points classified as red): 57%
overall error (average): 42.5%
```
(Note: exact error numbers might vary through executions)

---

As an attempt to further improve the algorithm, we will follow a common practice that falls within the scientific field of data mining: extracting additional features from the given dataset. Looking at the image above, we can see that most points of cluster 1 (red disks) appear near the center of the cartessian system O(0,0), while most blue points appear to be more widespread. So, we project our data in the 3-dimensional space, using the squared distance of each point from the center O as the third coordinate.

In other words, in the new coordinate system, each point is written as {X<sub>i</sub>, ||X<sub>i</sub>||<sup>2</sup>} = (x<sub>i</sub>, y<sub>i</sub>, x<sub>i</sub><sup>2</sup> + y<sub>i</sub><sup>2</sup>), where:
- X<sub>i</sub> = (x<sub>i</sub>, y<sub>i</sub>) are the original coordinates and 
- ![Euclidean Distance](https://latex.codecogs.com/svg.latex?\inline&space;\{||X_i||^2}=\left(\sqrt{(x_i-0)^2+(y_i-0)^2}\right)^2={x_i^2+y_i^2}) is the distanse of point x<sub>i</sub> from O(0,0).


The result this time (file 3b.py) is presented below, with the exact same notation used.
![Data Clustering v2](https://github.com/FireCoder-N/machine-learning/blob/main//K-means-clustering/3b.png?raw=true)

As expected and observed 'by inspection', using this method, the error is dramatically improved and its exact values in this particular example are:
```
error 1 (red points classified as blue): 1% (!)
error 2 (blue points classified as red): 69%
overall error (average): 35%
```
