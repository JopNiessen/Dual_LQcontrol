"""
Linear Quadratic system with unknown bias

by: J. Niessen
last update: 2022.10.13

"""

import numpy as np


class LQsystem:
    def __init__(self):
        self.v = .5
        self.R = 1
        self.F = 0
        self.G = 1
        self.b_vector = np.array([-1, 1])

    def x_update(self, x, u, b, xi):
        """
        Function    x(t+1) = x(t) + bu + dw
        :param x: position
        :param u: control
        :param b: bias
        :param xi: error
        :return: position (x) in the next timestep
        """
        return x + b*u + xi

    def theta_update(self, x_old, x_new, theta, u):
        """
        Function    theta(t+1) = theta(t) + (x(t+1) - x(t))u/v
        :param x_old: position in current timestep
        :param x_new: position in next timestep
        :param theta: belief state
        :param u: control
        :return: believe state in next timestep
        """
        return theta + (x_new - x_old)*u/self.v

    def bias_prob(self, b, theta):
        """
        Function    p(b|theta) = 1/(1 + e^(-2*b*theta))
        :param b: bias
        :param theta: belief state
        :return: probability of bias (b) given belief (theta)
        """
        return (1 + np.exp(-2*b*theta))**(-1)

    def normal(self, xi):
        """
        Calculate p(xi) = N(xi|0,v)
        :param xi: error term
        :return: probability density of error
        """
        return np.exp(-xi**2/(2*self.v))/(self.v*np.sqrt(2*np.pi))