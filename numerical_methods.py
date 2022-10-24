"""
Numerical optimization for LQ system with unknown bias and noise

by: J. Niessen
last update: 2022.10.13
"""

import numpy as np


class LQ_NumericalOptimizer:
    def __init__(self, system, T=30):
        """
        Numerical optimization by dynamic programming. Discretization over position, belief and time.
        Numerical integration over biases and noise.
        :param system: LQ system
        :param T: Finite time
        """
        # System parameters
        self.stm = system
        self.T = T
        self.t_vector = np.linspace(0, self.T, self.T+1).astype(np.int)

        # Discretized spaces
        self.xi_vector = np.linspace(-1, 1, 10)*10**np.sqrt(self.stm.v)     # Error terms
        self.xi_dist = self.stm.normal(self.xi_vector)                      # Probability of each error term
        self.xi_normalization = np.sum(self.xi_dist)                    # Normalization over discrete error distribution

        # X is distributed linearly over interval [-9,9]
        self.x_amp, self.x_dim = 9, 20
        self.x_vector = np.linspace(-self.x_amp, self.x_amp, self.x_dim)

        # theta is distributed nonlinear over interval [-13, 13]
        self.theta_amp, self.theta_dim = 13, 50
        self.theta_vector = np.linspace(-1, 1, self.theta_dim)**3 * self.theta_amp

        # HJB (optimal cost-to-go) matrix
        self.SolMat = np.zeros((self.x_dim, self.theta_dim, self.T + 1))

        """
        Gradient descent parameters
        """
        self.lr = .2            # learning rate
        self.tol = 10 ** (-2)   # tolerance
        self.max_iter = 100     # max number of iterations
        self.du = .5            # step size control

        # Calculate cost-to-go backward in time
        self.dynamic_programming()

    def dynamic_programming(self):
        """
        Calculates optimal cost-to-go (HJB eq) backward in time in discrete position, belief and time space
        :return: optimal cost-to-go. 3D matrix in ['position', 'belief', 'time']
        """
        for t in self.t_vector[::-1]:
            for i_theta, theta in enumerate(self.theta_vector):
                for i_x, x in enumerate(self.x_vector):
                    coord = np.array([x, theta])
                    self.SolMat[i_x, i_theta, t] = self.HJB(coord, t)

    def HJB(self, coord, t):
        """
        Calculates the cost-to-go (J) over optimal control (u*)
        :param coord: full state [x, theta]
        :param t: time
        :return: cost-to-go under the optimal control law J*(x(t), t)
        """
        x, theta = coord
        if t == self.T:
            return self.stm.F*x**2
        else:
            u_star = self.gradient_descent(coord, 0, t)
            return self.bellman_eq(coord, u_star, t)

    def bellman_eq(self, coord, u, t):
        """
        Function    J(x(t), u(t), t) = Ru^2 + sum_b( p(b|theta) sum_xi ( p(xi|0,v) * Gx(t+1)^2 * J(x(t+1), u(t+1), t) ))
        :param coord: full state [x, theta]
        :param u: control
        :param t: time
        :return: Cost_to_go under policy u, J(x(t), u(t), t)
        """
        x, theta = coord
        h = self.stm.R * u ** 2
        for b in self.stm.b_vector:
            x_new = self.stm.x_update(x, u, b, self.xi_vector)
            theta_new = self.stm.theta_update(x, x_new, theta, u)
            h += self.stm.bias_prob(b, theta) * (self.xi_dist @ (
                    self.stm.G * x_new ** 2 + self.jacobian_future(x_new, theta_new, t + 1))) / self.xi_normalization
        return h

    def gradient_descent(self, coord, u0, t):
        """
        Gradient descent
        :param coord: full state [x, theta]
        :param u0: initial control
        :param t: time
        :return: control (u) that minimizes the cost function (J)
        """
        u = u0
        for i in range(self.max_iter):
            gradient = self.numerical_gradient(coord, u, t)
            u -= self.lr * gradient
            if gradient <= self.tol:
                break
        return u

    def numerical_gradient(self, coord, u, t):
        """
        Approximate gradient
        :param coord: full state [x, theta]
        :param u: control
        :param t: time
        :return: Approximate gradient (J(u - delta(u)) / delta(u))
        """
        return (self.bellman_eq(coord, u + self.du, t) - self.bellman_eq(coord, u - self.du, t)) / (2 * self.du)

    def jacobian_future(self, x, theta, t):
        """
        Get Jacobian in the next time step (J_{t+1} (x(t+1), u(t+1), (t+1))
        :param x: position
        :param theta: belief
        :param t: time
        :return: Cost to go in a future state J(x(t+1), u(t+1), (t+1)
        """
        # Transform state values to closest matrix indexes
        x = self.transformation_x_to_idx(x)
        theta = self.transform_theta_to_idx(theta)

        coord = np.vstack([x, theta])

        lst = []
        for (i_x, i_theta) in coord.T:
            lst.append(self.SolMat[i_x, i_theta, t])
        return np.array(lst)

    def transformation_x_to_idx(self, x):
        """
        Transforms position (x) to closest idx of cost-to-go matrix
        :param x: position
        :return: index
        """
        x = (x + self.x_amp) / (2*self.x_amp)
        x = (self.x_dim - 1) * self.bind(x)
        return (x + .5).astype(np.int)

    def transform_theta_to_idx(self, theta):
        """
        Transforms belief (theta) to closest idx of cost-to-go matrix
        :param theta: belief
        :return: index
        """
        sign = np.sign(theta)
        idx = (self.theta_dim - 1)*(self.bind((abs(theta)/self.theta_amp)**(1/3)) * sign + 1) / 2
        return idx.astype(np.int)

    def bind(self, arr):
        """
        Binds all normalized values to range [0,1]
        :param arr: array of integers or floats
        :return: array of integers or floats, bound to range [0,1]
        """
        arr[arr < 0] = 0
        arr[arr > 1] = 1
        return arr








