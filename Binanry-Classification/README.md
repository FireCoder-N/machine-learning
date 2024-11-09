# Binary Classification (A Baysian + Neural Networks Approach)

This repository showcases the application of Bayesian decision theory and neural networks in solving binary classification tasks. The exercise explores the derivation of the Bayes optimal classifier, its implementation, the creation of a neural netrork binary classifier and a comparative analysis between them.

## Walkthrough
This file is a brief description of and introduction to binary classifiers. The process of the project is the following:
|#|Question| Code File(s) |Notes|
|-|--------|:------------:|-----|
|1.| Find the optimal Bayes Classifier to decide between hypotheses $H_0$ and $H_1$| - | Theoretical calculations are presented in detail in technical reports (report_en.pdf and report_gr.pdf)|
|2.| Construct a dataset where the samples (e.g. [x₁, x₂]) follow the distributions of the 2 hypotheses.|ex1.py| |
|3.| Calculate the optimal error as calculated in step 1.|ex1.py|
|4.| Construct a neural network by hand, **without** using any artificial intelligence libraries (e.g. tensorflow, pytorch) to classifiy the data| ex2.py | Once again, all calculations for designing the neural network with derrivates are detailed in both report files|
|5.| Apply gained knowledge as well as designed neural network (after modifications) to any binary classification problem, for example to diversify handwritten characters '0' and '8' from the MNIST dataset.| ex3.py|For this part different apporaches (e.g. activation functions) where followed, detailed in the report files. This step was only partially implemented.


## Introduction

The objective is to classify pairs of independent random variables under two competing hypotheses, $H_0$ and $H_1$. As an example, under  $H_0$, the variables are normally distributed with a mean of 0 and variance of 1. Under $H_1$, the variables follow a mixture of two Gaussians with equal probabilities. The exercise involved deriving the Bayes optimal test, evaluating its performance through simulation, and comparing it with a neural network's classification accuracy, after designing the neural network from scratch.

## Bayes Optimal Classification

### Theoretical Foundation

Bayesian decision theory provides a framework for constructing an optimal classifier based on the likelihood ratio of the two hypotheses. The goal is to minimize the classification error by comparing the likelihoods of the data under each hypothesis.

The derived Bayes classifier can be expressed as:

$$ 
\text{ln} \left( \frac{f_1(x_1) \cdot f_1(x_2)}{f_0(x_1) \cdot f_0(x_2)} \right) \gtrless 0
$$

Where:
- $f_0(x)$ represents the probability density function under $H_0$.
- $f_1(x)$ is the probability density function under $H_1$, representing a mixture of two Gaussians.

### Simulation and Performance Evaluation

To assess the effectiveness of the Bayes classifier, a large-scale simulation was conducted. We generated 1 million sample pairs $[x_1, x_2]$ under both $H_0$ and $H_1$, and applied the classifier to estimate the total classification error.

**Simulation Variables:**
- **$K$**: Number of sample pairs (set to $10^6$).
- **$f_{0_{samples}}$**: Data matrix for samples under $H_0$.
- **$f_{1_{samples}}$**: Data matrix for samples under $H_1$.

The total error was computed as the average of the misclassification rates under both hypotheses, providing a clear benchmark for the classifier's performance.

## Neural Network Classification

### Neural Network Implementation

In addition to the Bayes classifier, a neural network was implemented to tackle the same classification problem. The neural network was designed with a simple architecture: an input layer with 2 neurons, a hidden layer with 20 neurons, and an output layer with a single neuron. The network was trained using backpropagation and optimized with the cross-entropy loss function.

**Neural Network Details:**
- **Architecture:** 2-20-1 (2 input neurons, 20 hidden neurons, 1 output neuron).
- **Activation Functions:** ReLU for hidden layers, Sigmoid for the output.
- **Training Parameters:** 500 epochs, learning rate of 0.003.

### Comparative Analysis

After training, the neural network's classification performance can be compared against the Bayes optimal classifier. The analysis focused on the classification error rates, offering insights into the practical differences between a theoretically optimal classifier and a neural network trained on data.

## Conclusion

This exercise effectively demonstrated the principles of Bayesian classification and neural networks in the context of binary hypothesis testing. The Bayes optimal classifier provided a benchmark for classification performance, while the neural network highlighted the adaptability and potential of machine learning approaches. The comparison underscored the strengths of each method, illustrating their respective roles in modern data science.

