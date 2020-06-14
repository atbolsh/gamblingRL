import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

states = np.arange(101)

#Initial values.
#v = np.zeros(101)
#v[100] = 1.0    

def actionsAvailable(state):
    return np.arange(1, min((state, 100 - state)) + 1)

def expectedValue(state, action, ph, v):
    return ph*v[state + action] + (1 - ph)*v[state - action]

def newValue(state, ph, v):
    A = actionsAvailable(state)
    return max([expectedValue(state, action, ph, v) for action in A])

def bestAction(state, ph, v):
    A = actionsAvailable(state)
    return np.argmax(np.array([expectedValue(state, action, ph, v) for action in A])) + 1

def sweep(ph, v):
    D = 0
    for i in range(1, 100):
        vn = newValue(i, ph, v)
        D = max((D, vn - v[i]))
        v[i] = vn
    return D, v
 
def full(ph = 0.4, theta = 0.0):
    v = np.zeros(101)
    v[100] = 1.0
    D = 10
    trace = [deepcopy(v)]
    while D > theta:
        D, v = sweep(ph, v)
        trace.append(deepcopy(v))
    return trace    

trace = full()

def figs(vals, ph):
    plt.plot(states, vals)
    plt.xlabel("states")
    plt.ylabel("vals")
    plt.title("Valse, ph = " + str(ph))
    plt.show()
    plt.plot(states[1:-1], [bestAction(i, ph, vals) for i in range(1, 100)])
    plt.xlabel("states")
    plt.ylabel("Best action")
    plt.title("Best actions, ph = " + str(ph))
    plt.show()
    return None
    



