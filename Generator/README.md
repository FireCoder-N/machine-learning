# GAN
This repository contains a program that generates handwritten images of the digit `8` using a generative model.

## Introduction
Generative Adversarial Networks (GANs) are a class of deep learning models consisting of a generator and a discriminator. The generator aims to create realistic data samples, while the discriminator tries to distinguish between the generated samples and real data. The two models are trained together in a competitive manner, leading to the generation of realistic and high-quality samples.

In this project, a generator model is employed to generate handwritten images of the digit 8.

## Description
The generator takes as input a 10x1 vector, Z, whose elements follow the Gaussian distribution.
The second (hidden) layer has dimension of 128 and is activated by the ReLU function.
Finally, the output layer contains 784 neurons, so that the output 1784x1 vector can be reshaped to get the desired 28x28 image of a handwritten digit.

Using mathematic notation:

```
W₁ = A₁ * Z + B₁
Z₁ = max{W₁, 0} (ReLU)
W₂ = A₂ * Z + B₂
X  = 1/(1+exp(W₂)) 
```

where:
- Z is the input vector with dimensions 10x1,
- A<sub>1</sub> is the weight matrix and B<sub>1</sub> the bias vector with dimensions 128x10 and 128x1 respectively.
- Z<sub>1</sub> is the output of the hidden layer, with dimensions 128x1
- A<sub>2</sub> is the weight matrix and B<sub>2</sub> the bias vector with dimensions 784x128 and 784x1 respectively.
- X is the output vector with dimensions 784x1, representing a 28×28 image with a handwritten digit 8.
- `*` denotes matrix multiplication.

The model is already trained, with the matrices A<sub>1</sub>, B<sub>1</sub> and A<sub>2</sub>, B<sub>2</sub> given in the file `data21.mat`.


## Execution
The code of the file `ex21.py` handles opening the .mat file using scipy (Reference: https://stackoverflow.com/questions/874461/read-mat-files-in-python), creates a class for the generator and then runs the model 100 times (with a different input Z each time) to generate 100 realizations of the digit `8`.


## Results

The result of generating 100 handwritten digit 8 images is presented in the image below:

![Generated Digits](generated_digits.png)

Each row of the 10×10 matrix corresponds to a handwritten digit 8.
