# Datengenerierung Federpendel
import numpy as np
import csv
from NN import Neural_Network_One_Layer

np.random.seed(42)  # reproduzierbar
scale = 1.0

# Simulationsparameter
number_Trajectories  = 200
number_time_points = 500
time_Grid = np.linspace(0, 1, number_time_points)
dt       = time_Grid[1] - time_Grid[0]

# f(x,u) = u+x+1
def f(state, u):
    return state+u+1

data = []

for j in range(number_Trajectories):
    # Eine Zufällige Steuerung
    #--------------------------------------------------------------
    u= Neural_Network_One_Layer(n=10, input_dim=1, output_dim=1)

    # Zufälliger Startwert x0 in [-1.5,1.5]
    #------------------------------------------------------------
    #state = np.random.uniform(-1.5, 1.5, 1)   
    state = np.random.randn(1)
    for i in range(number_time_points):
        u_val = scale * u.ForwardPass(np.array([time_Grid[i]]))[0]
        # RK4 zum nächsten Zustand
        k1 = f(state,           u_val)
        k2 = f(state + dt/2*k1, u_val)
        k3 = f(state + dt/2*k2, u_val)
        k4 = f(state + dt*k3,   u_val)
        state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

        # Spalten: id, t, x, u
        #----------------------------------------------------
        data.append([j, time_Grid[i], state[0], u_val])

np.savetxt("OCP_dynamic_data.csv", data, delimiter=",", comments="", fmt="%.6f")