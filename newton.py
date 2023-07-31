# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: %(Elis)
"""
import numpy as np
import matplotlib.pyplot as plt

""" Functions and their derivatives"""
def function(xy):
    return np.asarray([xy[0]**3 - 3 * xy[0] * (xy[1]**2) - 1,
                       3 * (xy[0]**2) * xy[1] - xy[1]**3])

def derivative(self, xy):
    return np.asarray([[3 * (xy[0]**2 - xy[1]**2), -6*xy[0]*xy[1]],
                       [6*xy[0]*xy[1], 3 * (xy[0]**2 - xy[1]**2)]])

def function2(x):
    return np.asarray([x[0]**3 - 3*x[0]*x[1]**2 - 2*x[0] - 2,
                       3*x[1]*x[0]**2 - x[1]**3 - 2*x[1]]) 

def function3(x):
    return np.asarray([x[0]**8 - 28*(x[0]**6)*x[1]**2 + 70*(x[0]**4)*x[1]**4 + 15*x[0]**4 - 28*(x[0]**2)*x[1]**6 - 90*(x[0]**2)*x[1]**2 + x[1]**8 + 15*x[1]**4 - 16,
                       8*(x[0]**7)*x[1] - 56*(x[0]**5)*x[1]**3 + 56*(x[0]**3)*x[1]**5 + 60*(x[0]**3)*x[1] - 8*x[0]*x[1]**7 - 60*x[0]*x[1]**3])
    

class Fractal2D:
    """ Class variables"""
    zeroes = []
    tolerance = 0.01
    convergence_limit = 0.01
    iteration_limit = 20
    
    """ Constructor"""
    def __init__(self, function, derivative = None):
        self.function = function
        self.standard_method = Fractal2D.newtons_method
        self.standard_plot = Fractal2D.get_point_index
        if derivative is None:
            derivative = Fractal2D.numerical_derivative
        self.derivative = derivative
    
    """ Calculations"""
    def newtons_method(self, guess):
        iteration = 0
        while iteration <  Fractal2D.iteration_limit:
            adjustment = np.matmul(np.linalg.inv(self.derivative(self, guess)), self.function(guess))
            guess = guess - adjustment
            if np.sum(np.abs(adjustment)) < Fractal2D.convergence_limit:
                return (guess, iteration)
            iteration += 1
        return None
    
    
    def simplified_newtons_method(self, guess):
        iteration = 0
        jacobian = np.linalg.inv(self.derivative(self, guess))
        while iteration <  Fractal2D.iteration_limit:
            adjustment = np.matmul(jacobian, self.function(guess))
            guess = guess - adjustment
            if np.sum(np.abs(adjustment)) < Fractal2D.convergence_limit:
                return (guess, iteration)
            iteration += 1
        return None
    
    
    def numerical_derivative(self, guess):
        h = 0.00001
        xyh = self.function([guess[0] + h, guess[1]]) - self.function(guess)
        yxh = self.function([guess[0], guess[1] + h]) - self.function(guess)
        return np.asarray([[xyh[0]/h, yxh[0]/h], [xyh[1]/h, yxh[1]/h]])
        
    """ Plot support methods"""
    def get_point_index(self, initial_point):
        zero = self.standard_method(self, initial_point)
        if zero is None:
            Fractal2D.zeroes.append(None)
            return -1
        else:
            root = zero[0]
            tolerance = 0.0001
            for index, found_root in enumerate(Fractal2D.zeroes):
                if found_root is None or found_root[0] > root[0] + tolerance or found_root[1] > root[1] + tolerance or found_root[0] < root[0] - tolerance or found_root[1] < root[1] - tolerance:
                    continue
                return index
            index = len(Fractal2D.zeroes)
            Fractal2D.zeroes.append(root)
            return index
        
        
    def get_point_iterations(self, initial_point):
        zero = self.standard_method(self, initial_point)
        if zero is None:
            return Fractal2D.iteration_limit
        else:
            return zero[1]
    
    """ Plotting and meshing"""
    def plot(self, plot_index = True, use_simplified = None):
        if use_simplified:
            self.standard_method = Fractal2D.simplified_newtons_method
        if not plot_index:
            self.standard_plot = Fractal2D.get_point_iterations
        
        a, c = (-10, -10)
        b, d = (10, 10)
        N = 100
        x = np.linspace(a, b, N)
        y  = np.linspace(c, d, N)
        X, Y = np.meshgrid(x, y, indexing='ij')
        A = np.empty((N,N))
        for i in range(N):
            for j in range(N):
                A[i][j] = self.standard_plot(self, np.asarray([X[i][j], Y[i][j]]))
        print(A[80][50])
        print(A[20][20])
        print(A[20][80])
        print(A[79][1])
        plt.pcolor(A)
        plt.show()
        
    
fractal = Fractal2D(function3)
Fractal2D.plot(fractal)

        
    