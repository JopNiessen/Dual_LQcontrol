"""
Controllers for 2-dimensional LQ systems

by J. Niessen
created on: 2022.10.24
"""

import numpy as np
import numerical_methods2D


class LQG_Agent:
    def __init__(self, system, known_param):
        """
        Optimal controller for 2-dimensional LQ system through numerical optimization
        :param system: 2-dimensional LQ system
        :param known_param: known parameters [dict]
        """
        self.sys = system

        """System parameters"""
        self.A = known_param['A']
        self.C = known_param['C']
        self.C_inv = np.linalg.inv(self.C)
        self.v = known_param['v']
        self.w = known_param['w']
        self.T = known_param['T']

        """Agent state"""
        self.x_est = self.estimate_x()
        self.belief = 0
        self.B_est = None
        self.u = None
        self.t = 0

        """State memory"""
        self.z = {'t':[self.t], 'pos':[self.x_est[0]], 'vel':[self.x_est[1]], 'control':[], 'ExpBias':[0], 'cost':[]}

        """Possible values of unknown system parameter"""
        b_one = np.array([0, 1])*(-1)
        b_two = np.array([0, 1])
        self.B_space = np.stack((b_one, b_two), axis=0)

        """Numerical optimizer"""
        self.opt = None

    def run(self, perturb=list()):
        """Run agent through environment"""
        for i in range(self.T-1):
            if i in perturb:
                self.sys.B = self.sys.B * (-1)
            self.optimal_control()
        self.z['control'].append(0)
        self.z['cost'].append(self.sys.final_cost(self.sys.x))

    def estimate_x(self):
        """
        Observe current state
        :return: state observation (y)
        """
        return self.C_inv @ self.sys.observe()

    def estimate_theta(self, x_old, x_new, u): #PLACEHOLDER: this function now only updates theta1
        """
        Estimate belief
        :param x_old: previous observation
        :param x_new: current observation
        :param u: control
        :return: current belief (theta)
        """
        return self.belief + (x_new[1] - x_old[1]) * u[1] / self.w[1,1]

    def estimate_b(self, func_prob_b):
        """
        Estimate bias
        :param func_prob_b: Probability distribution of bias (B)
        :return: E[B] (expected value of the bias)
        """
        return self.B_space @ func_prob_b(self.B_space, self.belief)

    def optimal_control(self):
        """Finding optimal policy (u*) and exerting the control on the system"""
        u_star, J_star = self.opt.gradient_descent(self.x_est, self.belief, self.t)
        x_old = self.x_est
        cost = self.sys.update(u=u_star, info=True)
        self.t += 1
        self.x_est = self.estimate_x()
        self.belief = self.estimate_theta(x_old, self.x_est, u_star)
        exp_bias = 0
        for B in self.B_space:
            exp_bias += B * numerical_methods2D.prob_b(B, self.belief)
        self.add_info(self.t, self.sys.x[0], self.sys.x[1], u_star[1], exp_bias[1], cost)

    def add_info(self, t, x0, x1, u, Eb, J):
        """
        Memory update: add current (state) values to memory
        :param t: time
        :param x0: state variable 0
        :param x1: state variable 1
        :param u: control
        :param Eb: expected bias
        :param J: cost-to-go
        """
        self.z['t'].append(t)
        self.z['pos'].append(x0)
        self.z['vel'].append(x1)
        self.z['control'].append(u)
        self.z['ExpBias'].append(Eb)
        self.z['cost'].append(J)

    def reset(self, x0):
        """Reset system"""
        self.sys.x = x0

        """Reset agent state"""
        self.x_est = self.estimate_x()
        self.belief = 0
        self.B_est = None
        self.u = None
        self.t = 0

        """State memory"""
        self.z = {'t':[self.t], 'pos':[self.x_est[0]], 'vel':[self.x_est[1]], 'control':[], 'ExpBias':[0], 'cost':[]}


class BenchmarkAgent:
    def __init__(self, sys, T):
        """
        Runs system with no control
        :param system: 2-dimensional LQ system
        :param T: end time
        """
        self.T = T
        self.sys = sys

        self.z = {'t': [0], 'pos': [sys.x[0]], 'vel': [sys.x[1]], 'control':[0], 'ExpBias':[np.nan], 'cost': []}

    def run(self):
        """Run agent through environment"""
        for t in range(1, self.T):
            cost = self.sys.update(info=True)
            x = self.sys.x
            self.add_info(t, x[0], x[1], cost)
        self.z['cost'].append(self.sys.final_cost(x))

    def add_info(self, t, x0, x1, J):
        """
        Memory update: add current (state) values to memory
        :param t: time
        :param x0: state variable 0
        :param x1: state variable 1
        :param J: cost-to-go
        """
        self.z['t'].append(t)
        self.z['pos'].append(x0)
        self.z['vel'].append(x1)
        self.z['cost'].append(J)
        self.z['control'].append(0)
        self.z['ExpBias'].append(np.nan)


