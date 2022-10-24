"""
Methods for training and running agents trough environment

by J. Niessen
last update: 2022.10.24
"""

import system2D
import agents2D
import numerical_methods2D


def run_benchmark(x0, T):
    """
    Run benchmark through LQ system
    :param x0: initial position
    :param T: end time
    :return: [agent, trial results (z)]
    """
    sys = system2D.LQG_system(x0, T=T)
    bm = agents2D.BenchmarkAgent(sys, T)
    bm.run()
    return bm, bm.z


def run_optimal_agent(x0, T):
    """
    Train and run numerical optimal agent in LQ system
    :param x0: initial position
    :param T: end time
    :return: [agent, optimal cost-to-go tensor (J*), trial results (z)]
    """
    sys = system2D.LQG_system(x0, T=T)
    agt = agents2D.LQG_Agent(sys, sys.known_param)
    agt.opt = numerical_methods2D.LQG_NumericalOptimizer(sys.known_param, agt.B_space)
    agt.opt.run()
    agt.run()
    return agt, agt.opt.SolMat, agt.z

