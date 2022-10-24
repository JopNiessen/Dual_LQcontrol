"""
Agent finding running numerical optimal control in LQ system

by: J. Niessen
last update: 2022.10.13
"""

import numpy as np


class LQagent:
    def __init__(self, x0, optimizer, b=np.random.choice([-1, 1]), theta0=0, bias_perturb=None):
        """
        :param x0: initial position
        :param optimizer: numerical optimization methods earlier computed cost-to-go
        :param b: true bias term (only used for state updates)
        :param theta0: initial belief
        """
        self.x = x0
        self.theta = theta0
        self.t_vector = optimizer.t_vector[:-1]
        self.z = np.array([0, x0, theta0, None])
        self.opt = optimizer
        self.bias_perturb = bias_perturb

        self.b_true = b

    def run(self):
        """
        Runs agent through the system following numerical optimal control
        :return: None
        """
        for t in self.t_vector:
            if t == self.bias_perturb:
                self.b_true = -self.b_true
            u_star = self.state_update(t)
            self.z = np.vstack((self.z, np.array([t+1, self.x, self.theta, u_star])))

    def state_update(self, t):
        """
        Find optimal control u* in the current timestep and updates the full state
        :param t: time
        :return: optimal control (u*)
        """
        coord = np.array([self.x, self.theta])
        u_star = self.opt.gradient_descent(coord, 0, t)
        x_old = self.x
        xi = np.random.normal(0, self.opt.stm.v)
        self.x = self.opt.stm.x_update(x_old, u_star, self.b_true, xi)
        self.theta = self.opt.stm.theta_update(x_old, self.x, self.theta, u_star)
        return u_star
