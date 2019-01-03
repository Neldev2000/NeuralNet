import numpy as np
import random

def sigmoid( z):
        return 1.0 / (1.0 + np.exp(-z))
def sigmoid_prime(z):
        return sigmoid(z) * (1 - sigmoid(z))

def cost_derivative( output, y):
        #output based on mean square error function
        return (output-y)

class Neural_Network():
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes

        self.biases = [np.random.rand(x,1) for x in sizes[1:]]
        self.weights = [np.random.rand(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        pass
    
    
    
    def feed_forward(self, a):

        for w,b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(a,w) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data = None):
        """ 
            - Training data is a list of tuples (x,y) where x is the value to train and y is the exact result of that value.
            - eta: learning rate.
            - Epochs and mini_batch_size are self-explanatory.
            - Test data: Test data to compare while training
        """

        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0,n,mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            
            if test_data:
                print("Epoch {} {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete.".format(j))
        
        pass
            

    def evaluate(self, data):
        return 0

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nb, delta_nw = self.backprop(x,y)

            nabla_b += delta_nb
            nabla_w += delta_nw

        self.biases -= eta/len(mini_batch)*nabla_b
        self.weights -= eta/len(mini_batch)*nabla_w

        pass


    def backprop(self, x,y):

        """
            Return a tuple (nabla_b , nabla_w) representing the gradient of the 
            C_x cost function
        """
        nabla_b = [ np.zeros(b.shape) for b in self.biases ]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        #feedforward
        activation = x
        activations = [x] # list of all the activations
        zs = [] #list of all the z's values

        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        #Backward pass
        delta = cost_derivative(activations[-1],y) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(activations[-2].T, delta)

        for l in range(2,self.num_layers):
            z = zs[-1]
            sp = sigmoid_prime(z)

            delta = np.dot(self.weights[-l+1].T,delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)

        return nabla_b, nabla_w
    
    
    #End of the class
    pass

net = Neural_Network([784,60,10])

