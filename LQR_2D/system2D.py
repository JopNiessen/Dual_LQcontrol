"""
2-Dimensional Linear Quadratic (LQ) system with gaussian noise

by J. Niessen
created on: 2022.10.24
"""

import numpy as np

# For now, it uses timesteps (dt) of 1 sec. With x1 = 'velocity in (m/s)'
class LQG_system:
    def __init__(self, x0, b=np.random.choice([-1, 1]), k=.2, dt=1, T=4):
        """
        This class describes a 2 dimensional linear dynamical system with gaussian noise

        :param x0: initial state
        :param b: bias term
        :param k: friction
        :param dt: time step size
        :param T: end time
        """
        self.x = x0
        dim = len(x0)

        """System parameters"""
        self.A = np.identity(dim) + np.array([[0, dt], [0, -k]])
        self.B = np.array([0, b])
        self.C = np.identity(dim)
        self.v = np.array([[.5, 0], [0, .5]])
        self.w = np.array([[.001, 0],[0, .5]])

        """Cost parameters"""
        self.F = np.array([[1, 0], [0, 0]])
        self.G = np.array([[1, 0], [0, 0]])
        self.R = np.array([[1, 0], [0, 1]])
        self.T = T

        """Fully observable parameters"""
        self.known_param = {'dim':dim, 'A':self.A, 'C':self.C, 'v':self.v, 'w':self.w, 'F':self.F, 'G':self.G, 'R':self.R, 'T':self.T}

    def update(self, u=np.array([0, 0]), info=False):
        """
        Update state (x) according to: x(n+1) = Ax(n) + Bu(n) + Wxi
        :param u: control (zero if not assigned)
        :param info: [boolean] determines if cost is returned
        :return: marginal cost
        """
        self.x = self.A @ self.x + self.B * u + self.w @ np.random.normal(0, 1, 2)
        if info:
            return self.cost(self.x, u)

    def observe(self):
        """
        Observe the state (x) according to: y(n) = Cx(n) + Vxi
        :return: state observation (y)
        """
        return self.C @ self.x + self.v @ np.random.normal(0, 1, 2)

    def cost(self, x, u=np.array([0, 0])):
        """
        (Marginal) cost
        :param x: state
        :param u: control
        :return: marginal cost
        """
        return x @ self.G @ x + u @ self.R @ u

    def final_cost(self, x):
        """
        Cost in final timestep (t=T)
        :param x: state
        :return: end cost
        """
        return x @ self.F @ x

    def reset(self, x0):
        """
        Reset state
        :param x0: initial state
        """
        self.x = x0



def update_x(A, B, x, u, xi):
    """
    Calculate state (x) in next timestep
    :param A: System matrix
    :param B: Bias
    :param x: state
    :param u: control
    :param xi: noise
    :return: state in next timestep x(n+1)
    """
    return A @ x + B * u + xi


def update_theta(x_old, x_new, theta, u, cov_inv):
    """
    Calculate new belief
    :param x_old: previous state x(n-1)
    :param x_new: current state x(n)
    :param theta: previous belief theta(n-1)
    :param u: last control u(n-1)
    :param W: state covariance matrix
    :return: new belief theta(n)
    """
    #print((theta + (x_new - x_old) * u @ np.linalg.inv(cov)).shape)
    return theta + (x_new - x_old) * u @ cov_inv

