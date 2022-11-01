"""
Numerical methods for optimal control in a partly observable 2-dimensional LQ system

by J. Niessen
created on: 2022.10.24
"""

import numpy as np
from scipy.stats import multivariate_normal
from system2D import update_x, update_theta
from tqdm import tqdm

import multiprocessing


class LQG_NumericalOptimizer:
    def __init__(self, system_param, B_space, method='gradient_descent'):
        """
        Numerical optimization algorithm for a LQ system with unknown bias term
        :param system_param: Fully observable system parameters
        :param B_space: Possible bias terms
        """
        self.param = system_param
        self.t = None

        """System parameters"""
        self.dim = self.param['dim']
        self.A = self.param['A']
        self.C = self.param['C']
        self.W = self.param['w']
        self.V = self.param['v']

        self.cov_xi = self.W + self.A @ np.linalg.inv(self.C) @ self.V
        self.cov_theta = self.W + (self.A + np.identity(self.dim)) @ np.linalg.inv(self.C) @ self.V
        self.cov_theta_inv = np.linalg.inv(self.cov_theta)

        self.u0 = np.zeros(self.dim)

        """Discrete time"""
        self.T = self.param['T']
        self.t_vector = np.linspace(0, self.T, self.T + 1).astype(np.int)

        """Discrete gaussian noise"""
        xi0_cov, xi1_cov = self.cov_xi.diagonal()
        xi_pos = np.linspace(-1, 1, 10) * 10 ** np.sqrt(xi0_cov)
        xi_vel = np.linspace(-1, 1, 10) * 10 ** np.sqrt(xi1_cov)
        self.xi_matrix = np.transpose(np.meshgrid(xi_pos, xi_vel), (1, 2, 0))
        xi_dist = multivariate_normal.pdf(self.xi_matrix, np.array([0, 0]), self.cov_xi)
        self.xi_dist = xi_dist / np.sum(xi_dist)

        """Discrete space of unobserved bias term"""
        self.B_space = B_space

        """Discrete state space"""
        self.x_dim = 20
        self.pos_amp, self.pos_dim = 9, self.x_dim
        self.pos_vector = np.linspace(-1, 1, self.pos_dim) * self.pos_amp
        self.vel_amp, self.vel_dim = 9, self.x_dim
        self.vel_vector = np.linspace(-1, 1, self.vel_dim) * self.vel_amp
        self.x_amp = np.array([self.pos_amp, self.vel_amp])

        """Discrete belief space"""
        self.theta_amp, self.theta_dim = 13, 25
        self.theta_vector = np.linspace(-1, 1, self.theta_dim) ** 3 * self.theta_amp

        """HJB (optimal cost-to-go) matrix"""
        self.SolMat = np.zeros((self.pos_dim, self.vel_dim, self.theta_dim, self.T + 1))

        """
        Gradient descent parameters
        """
        self.method = method
        self.lr = .2
        self.tol = 10 ** (-2)
        self.max_iter = 20
        self.du = np.array([0, 1]) * 1

    def run(self, distributed_processing=True):
        """Calculate optimal cost-to-go (J*)"""
        if distributed_processing:
            self.distributed_run()
        else:
            self.dynamic_programming()

    def distributed_run(self):
        """
        Distributes optimization for the HJB equation for different initial conditions over multiple processes
        """
        ispace = np.meshgrid(range(0, self.theta_dim), range(0, self.pos_dim), range(self.vel_dim))
        ispace = np.array(ispace).reshape([3, -1]).T
        space = np.meshgrid(self.theta_vector, self.pos_vector, self.vel_vector)
        space = np.array(space).reshape([3, -1]).T
        space_lst = [[idx[0], idx[1], idx[2], item[0], item[1], item[2]] for idx, item in zip(ispace, space)]
        for t in tqdm(self.t_vector[::-1]):
            self.t = t
            p = multiprocessing.Pool()
            Jstar = p.map(self.update_Jstar, space_lst)
            for itheta, ipos, ivel, it, HJB in Jstar:
                self.SolMat[ipos, ivel, itheta, it] = HJB

    def update_Jstar(self, info):
        """
        Distributed optimization of HJB equation
        :param info: contains [belief idx, state[0] idx, state[1] idx, belief, state[0], state[1]]
        :return: [belief idx, state[0] idx, state[1] idx, time, optimal cost-to-go (J*)]
        """
        itheta, ipos, ivel, theta, pos, vel = info
        t = self.t
        HJB = self.HJB(np.array([pos, vel]), theta, t)  # Calculate optimal cost-to-go
        return itheta, ipos, ivel, t, HJB

    def dynamic_programming(self):
        """
        Calculate J* in discrete time, state and belief space backward in time through dynamic programming
        """
        for t in tqdm(self.t_vector[::-1]):
            for itheta, theta in tqdm(enumerate(self.theta_vector)):
                for ix, x in enumerate(self.pos_vector):
                    for ivel, vel in enumerate(self.vel_vector):
                        HJB = self.HJB(np.array([x, vel]), theta, t)  # Calculate optimal cost-to-go
                        self.SolMat[ix, ivel, itheta, t] = HJB

    def HJB(self, x, theta, t):
        """
        Calculate optimal cost-to-go (J*) through the Hamiltonian-Jacobian bellman equation
        :param x: state
        :param theta: belief
        :param t: time
        :return: optimal cost-to-go (J*)
        """
        if t == self.T:
            # At end-time, this returns the end-cost
            return x.T @ self.param['F'] @ x
        else:
            if self.method == 'gradient_descent':
                u_star, J_star = self.gradient_descent(x, theta, t)  # Establish the optimal control (u)
                return J_star
            else:
                pass

    def gradient_descent(self, x, theta, t):
        """
        Finding the control (u) that minimizes the cost-to-go (bellman equation) through discrete gradient descent
        :param x: state
        :param theta: belief
        :param t: time
        :return: approximate optimal control (u*)
        """
        J_star = np.inf
        u_star = 0
        # Looping over different initial control-values prevents convergence to local minima
        for u_loc in np.vstack([np.zeros(6), np.linspace(-3, 3, 6)]).T:
            # Apply gradient descent update until convergence is observed or max_iter time-steps
            for i in range(self.max_iter):
                gradient = self.numerical_gradient(x, theta, u_loc, t)
                u_loc -= self.lr * gradient
                if np.sum(gradient) <= self.tol:
                    break
            J_loc = self.bellman_eq(x, theta, u_loc, t)
            if J_loc < J_star:  # Update the (best) minimum found
                J_star = J_loc
                u_star = u_loc
        return u_star, J_star

    def numerical_gradient(self, x, theta, u,
                           t):  # PLACEHOLDER: this has to become a 2-D gradient, update both u0 and u1
        """
        Numerical gradient calculation
        :param x: state
        :param theta: belief
        :param u: control
        :param t: time
        :return: gradient
        """
        return (self.bellman_eq(x, theta, u + self.du, t) - self.bellman_eq(x, theta, u - self.du, t)) / (
                    2 * self.du[1])

    def bellman_eq(self, x, theta, u, t):
        """
        Bellman equation, calculates the cost-to-go
        :param x: state
        :param theta: belief
        :param u: control
        :param t: time
        :return: cost-to-go (J)
        """
        jac = u.T @ self.param['R'] @ u
        for b in self.B_space:
            x_new = update_x(self.A, b, x, u, self.xi_matrix)
            theta_new = update_theta(x, x_new, theta, u, self.cov_theta_inv)
            Pb = prob_b(b, theta)
            Rnew = np.sum(x_new @ self.param['R'] * x_new, axis=2)
            Jnew = self.get_J(x_new, theta_new, t + 1)
            EJ = np.sum(self.xi_dist * (Rnew + Jnew))
            jac += Pb * EJ
        return jac

    def get_J(self, x, theta, t):
        """
        Get (known, earlier calculated) cost-to-go from memory
        :param x: position
        :param theta: belief
        :param t: time
        :return: Cost to go in given time-step
        """
        # Transform state values to closest matrix indexes
        x = self.transformation_x_to_idx(x)
        theta = self.transform_theta_to_idx(theta)

        # Reshape for efficient calculation
        coord = np.dstack([x, theta])
        coord_shape = coord.shape[:2]
        coord = coord.reshape(-1, coord.shape[-1])

        # Get J* from the earlier established tensor in memory
        lst = self.SolMat[coord[:, 0], coord[:, 1], coord[:, 3], t]

        # Transform to original shape
        cost = np.array(lst).reshape(coord_shape)
        return cost

    def transformation_x_to_idx(self, x):  # PLACEHOLDER: can change this to estimation based on closest nodes
        """
        Transform state (x) to closest idx of cost-to-go matrix
        :param x: position
        :return: index
        """
        x = (x + self.x_amp) / (2 * self.x_amp)
        x = (self.x_dim - 1) * self.bind(x)
        x = (x + .5).astype(np.int)
        return x

    def transform_theta_to_idx(self, theta):
        """
        Transform belief (theta) to closest idx of cost-to-go matrix
        :param theta: belief
        :return: index
        """
        sign = np.sign(theta)
        idx = (self.theta_dim - 1) * (self.bind((abs(theta) / self.theta_amp) ** (1 / 3)) * sign + 1) / 2
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


def normal(x, mu, sigma):
    """
    Calculate p(x) = N(x|mu,sigma)
    :param x: variable
    :param mu: mean
    :param sigma: standard deviation
    :return: probability density of x
    """
    return np.exp((mu - x) ** 2 / (2 * sigma)) / (sigma * np.sqrt(2 * np.pi))


def prob_b(b, theta):
    """
    Function    p(b|theta) = 1/(1 + e^(-2*b*theta))
    :param b: bias
    :param theta: belief state
    :return: probability of bias (b) given belief (theta)
    """
    return (1 + np.exp(-2 * b[1] * theta)) ** (-1)
