"""
Numerical optimal control of LQ system with unknown bias

by: J. Niessen
last update: 2022.10.13
"""

import numpy as np
from numerical_methods import *
from system import *
from agent import *

import matplotlib.pyplot as plt


sys = LQsystem()
opt = LQ_NumericalOptimizer(sys, T=60)


fig, ax = plt.subplots(3, figsize=(10,8))
clr = ['b', 'r']
b_vct = [-1, 1]

for x0 in np.linspace(-9,9,10):
    bi = np.random.choice([0,1])
    agent = LQagent(7, opt, b=b_vct[bi], bias_perturb=30)
    agent.run()
    ax[0].plot(agent.z[:,0], agent.z[:,1], c=clr[bi])
    ax[1].plot(agent.z[:,0], agent.z[:,3], c=clr[bi])
    theta = np.array(agent.z[:,2]).astype(np.float)
    E_bias = sys.bias_prob(1,theta) - sys.bias_prob(-1,theta)
    ax[2].plot(agent.z[:,0], E_bias, c=clr[bi])
ylab = ['position (x)', 'control (u)', 'expected bias (hat{b})']

for i in range(3):
    ax[i].set_xlabel('t')
    ax[i].set_ylabel(ylab[i])
    ax[i].grid()

plt.show()


