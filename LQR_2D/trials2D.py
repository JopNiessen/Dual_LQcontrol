"""
Methods for training and running agents trough environment

by J. Niessen
last update: 2022.10.24
"""

import system2D
import controllers2D
import numerical_methods2D


def run_benchmark(x0, T):
    """
    Run benchmark through LQ system
    :param x0: initial position
    :param T: end time
    :return: [agent, trial results (z)]
    """
    sys = system2D.LQG_system(x0, T=T)
    bm = controllers2D.BenchmarkAgent(sys, T)
    bm.run()
    return bm, bm.z


def train_optimal_agent(x0, T, CE=False):
    """
    Train numerical optimal agent in LQ system
    :param x0: initial position
    :param T: end time
    """
    sys = system2D.LQG_system(x0, T=T)
    agent = controllers2D.LQG_Agent(sys, sys.known_param)
    agent.opt = numerical_methods2D.LQG_NumericalOptimizer(sys.known_param, agent.B_space, CE=CE, CA=False)
    agent.opt.run(distributed_processing=True)
    return agent



