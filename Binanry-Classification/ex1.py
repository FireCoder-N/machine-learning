import numpy as np

class Test:
    def __init__(self, K=10**6):
        self.K = K # Number of samples

        # following f₀ distribution
        self.f0_samples_x1 = np.random.normal(0, 1, K) # Generate K samples from N(0,1) for x₁
        self.f0_samples_x2 = np.random.normal(0, 1, K) # Generate K samples from N(0,1) for x₂
        self.f0_samples = np.vstack((self.f0_samples_x1, self.f0_samples_x2)).T # Stack the samples together, so that each row is a sample X=[x₁, x₂]

        # similarly, following f₁ distribution
        self.f1_samples_x1 = np.random.choice([-1, 1], size=K) # Generate K samples from {−1,1} for x₁
        self.f1_samples_x1 = np.random.normal(self.f1_samples_x1, 1) # then each sample x₁ takes its value from N(−1,1) or N(1,1), based on the previous step

        #same for x₂
        self.f1_samples_x2 = np.random.choice([-1, 1], size=K) 
        self.f1_samples_x2 = np.random.normal(self.f1_samples_x2, 1)

        self.f1_samples = np.vstack((self.f1_samples_x1, self.f1_samples_x2)).T # Stack the samples together

    def h(self, x):
        return np.log(0.5) + np.log( np.exp(np.square(x+1)/(-2)) + np.exp(np.square(x-1)/(-2)) ) + np.square(x)/2
    
    def Test(self):
        f0_test = self.h(self.f0_samples[:,0]) + self.h(self.f0_samples[:,1])

        f0_errors = np.count_nonzero((f0_test > 0) | ((f0_test == 0) & (np.random.rand(self.K) < 0.5)))
        f0_errors /= self.K

        f1_test = self.h(self.f1_samples[:,0]) + self.h(self.f1_samples[:,1])

        f1_errors = np.count_nonzero((f1_test < 0) | ((f1_test == 0) & (np.random.rand(self.K) < 0.5)))
        f1_errors /= self.K

        print("f0:", f0_errors, "f1:", f1_errors)
        print(0.5*f0_errors + 0.5*f1_errors)

    
# Compute the probability of error
if __name__ == "__main__":
    test = Test()
    test.Test()