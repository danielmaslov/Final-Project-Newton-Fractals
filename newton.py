# -*- coding: utf-8 -*-
"""
Created on %(date)s
@author: %(username)s
"""
from  numpy import *
from  matplotlib.pyplot import *

import numpy as np
import matplotlib.pyplot as plt

class fractal2D:
    def __init__(self, f, f_derivative=None, max_iterations=10000, threshold=1e-6):
        self.f = f
        self.f_derivative = f_derivative
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.zeroes = []  # List to store the found zeroes
        self.jacobian = None
    
    def newton_method(self, z):
        for i in range(self.max_iterations):
            if self.f_derivative is not None:
                f_prime = self.f_derivative(z)
            else:
                f_prime = self._numerical_derivative(z)

            if np.all(np.sum(np.abs(f_prime) < self.threshold)):
                break

            J_inv = np.linalg.inv(f_prime)
            z_new = z - np.dot(J_inv, self.f(z))
            if np.all(np.sum(np.abs(z_new - z)) < self.threshold):
                for idx, zero in enumerate(self.zeroes):
                    if np.linalg.norm(z_new - zero) < self.threshold:
                        return idx  # Return the index of the existing zero if within tolerance

                self.zeroes.append(z_new)  # Add the newly found zero to the list
                return len(self.zeroes) - 1  # Return the index of the newly found zero

            z = z_new

        return -1  # Return -1 if the algorithm did not converge



    
    
    
    def get_point_index(self, initial_point):
        
        z, num_iterations = self.newton_method(initial_point) 
        # Assuming newton_method returns (z, num_iterations)
    
        if z is not None:
            self.zeroes.append(None)
            return -1  # Return -1 if the algorithm did not converge
        else:
            for i, zero in enumerate(self.zeroes):
                if np.linalg.norm(z - zero) < self.threshold:
                    return i  # Return the index of the existing zero if within tolerance

            self.zeroes.append(z)  # Add the newly found zero to the list
            return len(self.zeroes) - 1  # Return the index of the newly found zero

       


    
    def simplified_newton_method(self, z):
        if self.jacobian is None:
            if self.f_derivative is not None:
                self.jacobian = self.f_derivative(z)
            else:
                self.jacobian = self._numerical_derivative(z)

        for i in range(self.max_iterations):
            if np.all(np.abs(self.jacobian) < self.threshold):
                break

            J_inv = np.linalg.inv(self.jacobian)
            z_new = z - np.dot(J_inv, self.f(z))

            if np.linalg.norm(z_new - z) < self.threshold:
                return z_new

            z = z_new

        return None  
    
    
    def plot(self, N, a, b, c, d, use_simplified=False):
      # Create a meshgrid of N^2 points in the set G = [a, b] x [c, d]
       x_vals = np.linspace(a, b, N)
       y_vals = np.linspace(c, d, N)
        
             
       X, Y = np.meshgrid(x_vals, y_vals, indexing = 'ij')
       
       A = np.empty((N, N), dtype=int)

# Iterate through each grid point (x0, y0) represented by X and Y
       for i in range(N):
           for j in range(N):
               index = self.newton_method(np.array([X[i, j], Y[i, j]]))

               A[i, j] = index

       

       
       plt.figure(figsize=(8, 6))
       plt.pcolor(X, Y, A, cmap='inferno')
       plt.colorbar(label='Index of Converged Zero')
       plt.xlabel('x')
       plt.ylabel('y')
       plt.title('Dependence of Newton\'s Method on Initial Vectors')
       plt.show()
        
    
    
    
    
    
    def _numerical_derivative(self, z):
        h = 1e-8
        df_dx = (self.f([z[0] + h, z[1]]) - self.f([z[0] - h, z[1]])) / (2 * h)
        df_dy = (self.f([z[0], z[1] + h]) - self.f([z[0], z[1] - h])) / (2 * h)
        return np.array([df_dx, df_dy])
    
    

    def generate_fractal(self, x_min, x_max, y_min, y_max, width, height):
        x_vals = np.linspace(x_min, x_max, width)
        y_vals = np.linspace(y_min, y_max, height)
        fractal = np.zeros((height, width))

        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                z = np.array([x, y])
                root = self.newton_method(z)
                fractal[i, j] = self._get_root_index(root)

        return fractal
   
        
    

# Example usage:
if __name__ == "__main__":
    
   
    def f(z):
        return np.array([z[0]**3 - 3*z[0]*z[1]**2 - 1, 3*z[0]**2*z[1] - z[1]**3])

    # Define the derivative of the function f
    def f_derivative(z):
        return np.array([[3*z[0]**2 - 3*z[1]**2, -6*z[0]*z[1]], [6*z[0]*z[1], 3*z[0]**2 - 3*z[1]**2]])
    
    # Initialize the fractal2D class
    #print(f_derivative([1,1]))
    #print(f([1,1]))
    
    solver = fractal2D(f, f_derivative, max_iterations=5000, threshold=1e-6)
    
    # Parameters for plotting
    N = 100  # Number of points in each direction
    a, b = -7, 7  # x range
    c, d = -4, 4  # y range
    use_simplified = False  # Set to True if you want to use simplified_newton_method

    # Plot using the plot method from the solver
    solver.plot(N, a, b, c, d, use_simplified)

    


    

    










