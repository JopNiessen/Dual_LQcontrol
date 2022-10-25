"""
Control for 2-dimensional Linear Quadratic (LQ) systems with unknown bias,
through numerical optimization of the Hamiltonian-Jacobian-Bellman (HJB) equation

by J. Niessen
last update: 2022.10.24
"""

import trials2D
import pickle

import numpy as np
import matplotlib.pyplot as plt

"""
Run optimal agent
"""
T = 6
x0 = np.array([10, 0])
agt = trials2D.train_optimal_agent(x0, T)
agt.run()

#file = open('saved/agent_T{}_v20221024.pickle'.format(T), 'wb')
#pickle.dump(agent, file)
#file.close()

M = agt.opt.SolMat
Z = agt.z

bm_agent, bm_Z = trials2D.run_benchmark(x0, T)

"""
Plot cost maps
"""
fig, ax = plt.subplots(2, 6, figsize=(8, 8))
ax = ax.flatten()
i = 0
for vel in [5, 9]:
    for time in [-1, -2, -3, -4, -5, -6]:
        ax[i].imshow(M[:,vel,:,time])
        ax[i].invert_yaxis()
        ax[i].set_title('time={0}, vel={1}'.format(time, vel))
        i += 1
plt.tight_layout()
plt.show()

"""
Plot trial
"""
items = ['pos', 'vel', 'control', 'ExpBias', 'cost']
labels = ['position (x)', 'velocity (v)', 'control (u)', 'E(bias)', 'cost']

plt.style.use('ggplot')
fig, ax = plt.subplots(len(items))
for i, item in enumerate(items):
    ax[i].plot(Z['t'], Z[item], label='numerical optimum')
    ax[i].plot(bm_Z['t'], bm_Z[item], label='no-control')
    ax[i].set_ylabel(labels[i])
    ax[i].set_xticks(np.arange(0, T, step=1 + int((T - 1) / 10)))
ax[0].axhline(0, c='black', linestyle='--', label='optimum')
ax[1].axhline(0, c='black', linestyle='--', label='optimum')
ax[0].legend(loc=1)
ax[-1].set_xlabel('time (t)')
plt.show()

