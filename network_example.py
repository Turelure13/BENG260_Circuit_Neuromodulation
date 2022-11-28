"""
An example of a network simulation

@author: Luka
"""

import matplotlib.pyplot as plt

from neuron_model import Neuron
from network_model import CurrentSynapse, ResistorInterconnection, Network
import numpy as np

# Define timescales
tf = 0
ts = 50
tus = 50 * 50

# Define circuit elements
n = 2  # number of neurons
neurons = []  # list of all neurons in the network


# g_t = 2.45*10**-2
# # Professor equation
# V_post = sol.y[3]
# V_pre = sol.y[0]
# S_of_V_pre = (1/2) * (1 + np.tanh((V_pre-voff)/voff))
# E_ini = -75
# I_synap = g_t * S_of_V_pre * (E_ini - V_post)
#for j in range(2):
#     # Define empty neurons and then interconnect the elements
#     neuron = Neuron()
#     R = neuron.add_conductance(1)
#     i1 = neuron.add_current(-2, 0, tf)  # fast negative conductance
#     i2 = neuron.add_current(2, 0, ts)  # slow positive conductance
#     i3 = neuron.add_current(-1.5, -1.5, ts)  # slow negative conductance
#     i4 = neuron.add_current(1.5, -1.5, tus)  # ultraslow positive conductance
#
#     neurons.append(neuron)

# This analysis is consistent with the biophysics
# of excitable neurons: sodium channel activation
# is fast and acts as a negative conductance close
# to the resting potential, whereas potassium channel
# activation is slow and acts as a positive conductance.
# 2.4, 1.89, 4
potassium = 1.89
sodium = 1.89
conduct = 1.89

for j in range(2):
     # Define empty neurons and then interconnect the elements
     neuron = Neuron()
     R = neuron.add_conductance(1)
     i1 = neuron.add_current(-potassium, 0, tf)  # fast negative conductance
     i2 = neuron.add_current(potassium, 0, ts)  # slow positive conductance
     i3 = neuron.add_current(-1.5, -1.5, ts)  # slow negative conductance
     i4 = neuron.add_current(1.5, -1.5, tus)  # ultraslow positive conductance

     neurons.append(neuron)

# HEALTHY NETWORK DOPAMINE
# First Neuron
# Define empty neurons and then interconnect the elements
#neuron = Neuron()
#R = neuron.add_conductance(1)
#i1 = neuron.add_current(2.45, 0, tf)  # fast negative conductance
#i2 = neuron.add_current(0, 0, ts)  # slow positive conductance
#i3 = neuron.add_current(0, 0, ts)  # slow negative conductance
#i4 = neuron.add_current(0, 0, tus)  # ultraslow positive conductance
#neurons.append(neuron)
## Second Neuron
## Define empty neurons and then interconnect the elements
#neuron = Neuron()
#R = neuron.add_conductance(1)
#i1 = neuron.add_current(2.45*10**-2, 0, tf)  # fast negative conductance
#i2 = neuron.add_current(0, 0, ts)  # slow positive conductance
#i3 = neuron.add_current(0, 0, ts)  # slow negative conductance
#i4 = neuron.add_current(0, 0, tus)  # ultraslow positive conductance
#neurons.append(neuron)


# Define the connectivity matrices
#g_inh = [[0, -0.075], [0, 0]]  # inhibitory connection neuron 1 -| neuron 2
#g_exc = [[0, 0], [0, 0]]  # excitatory connection neuron 1 <- neuron 2
#g_res = [[0, 0], [0, 0]]  # resistive connections

# Define the connectivity matrices
g_inh = [[0, .75], [0, 0]]  # inhibitory connection neuron 1 -| neuron 2
g_exc = [[0, 0], [0, 0]]  # excitatory connection neuron 1 <- neuron 2
g_res = [[0, 0], [0, 0]]  # resistive connections

voff = -1
inh_synapse = CurrentSynapse(-1, voff, ts)
exc_synapse = CurrentSynapse(+1, voff, ts)
resistor = ResistorInterconnection()

# Define the network
network = Network(neurons, (inh_synapse, g_inh), (exc_synapse, g_exc),
                  (resistor, g_res))

# Simulate the network
trange = (0, 20000)
#10000 = 4sec

# Define i_app as a function of t: returns an i_app for each neuron
# I_{synapse}(t) = g(t) S(V_{pre(t))[E_{excitatory} - V_{post}(t)]
i_app = lambda t: [-2.1, -2]


sol = network.simulate(trange, i_app)

# g_t = np.sin((0.0001)*np.pi*sol.t)
# g_t = (2.45*10**-12)
g_t = conduct*10**-9
# g_t = g_t/abs(np.min(g_t))
# I_syn = sol.y[3][0] + g_t*sol.y[3]
# I_syn = sol.y[3]
# Professor equation
V_post = sol.y[3]
V_pre = sol.y[0]
S_of_V_pre = (1/2) * (1 + np.tanh((V_pre-voff)/voff))
E_ini = -0.075
I_synap = g_t * S_of_V_pre * (E_ini - V_post)
volt = S_of_V_pre * (E_ini - V_post)

# ADHD
# g_t = (2.45*10**-12)/2
g_t = (conduct*10**-9)*0.3
V_post = sol.y[3]
V_pre = sol.y[0]
S_of_V_pre = (1/2) * (1 + np.tanh((V_pre-voff)/voff))
E_ini = -0.075
I_synap_ADHD = g_t * S_of_V_pre * (E_ini - V_post)

# ADHD with stimulant
# g_t = (2.45*10**-12)/2
g_t = ((conduct*10**-9)*0.3)+((conduct*10**-9)*0.3)*0.7
V_post = sol.y[3]
V_pre = sol.y[0]
S_of_V_pre = (1/2) * (1 + np.tanh((V_pre-voff)/voff))
E_ini = -0.075
I_synap_ADHD_stimu = g_t * S_of_V_pre * (E_ini - V_post)

# Plot simulation
# y[0] = neuron 1 membrane voltage, y[3] = neuron 2 membrane voltage
fig = plt.figure()
test_time = np.interp(sol.t, (sol.t.min(), sol.t.max()), (0, 9))
plt.plot(test_time, sol.y[0], test_time, sol.y[3])
plt.xlabel('Time (sec)')
plt.ylabel('Voltage (V)')
plt.legend(['neuron 1 membrane voltage', 'neuron 2 membrane voltage'])
plt.show()
fig.savefig('/Users/arthurlefebvre/PycharmProjects/BENG260_Circuit_Neuromodulation/plots/Volt_time.png', dpi=300)

fig = plt.figure()
plt.plot(test_time, I_synap)
plt.plot(test_time, I_synap_ADHD, alpha=0.5)
plt.plot(test_time, I_synap_ADHD_stimu, alpha=0.5)
plt.xlabel('Time (sec)')
plt.ylabel('Current (A)')
plt.legend(['I_synap without ADHD', 'I_synap with ADHD', 'I_synap with ADHD with stimu'])
plt.show()
fig.savefig('/Users/arthurlefebvre/PycharmProjects/BENG260_Circuit_Neuromodulation/plots/current_time.png', dpi=300)


fig = plt.figure()
test = np.zeros((np.size(sol.y[3]), 2))
test[:, 0] = sol.y[3]
test[:, 1] = I_synap
test2 = np.sort(test, axis=0)
plt.plot(test2[:, 0], test2[:, 1])
test = np.zeros((np.size(sol.y[3]), 2))
test[:, 0] = sol.y[3]
test[:, 1] = I_synap_ADHD
test2 = np.sort(test, axis=0)
plt.plot(test2[:, 0], test2[:, 1])
test = np.zeros((np.size(sol.y[3]), 2))
test[:, 0] = sol.y[3]
test[:, 1] = I_synap_ADHD_stimu
test2 = np.sort(test, axis=0)
plt.plot(test2[:, 0], test2[:, 1])
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.legend(['I_synap without ADHD', 'I_synap with ADHD', 'I_synap with ADHD with stimu'])
plt.show()
fig.savefig('/Users/arthurlefebvre/PycharmProjects/BENG260_Circuit_Neuromodulation/plots/current_volt.png', dpi=300)
