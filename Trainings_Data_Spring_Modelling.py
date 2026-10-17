# Datengenerierung Federpendel
import numpy as np
import csv
from NN import Neural_Network_One_Layer

np.random.seed(42)  # reproduzierbar
scale = 0.5

# Simulationsparameter
number_Trajectories  = 50
number_time_points = 500
time_Grid = np.linspace(0, 20, number_time_points)
dt       = time_Grid[1] - time_Grid[0]


#m, k = 0.1, 0.1   
m, k = 2.0 , 0.5
# f(x,u) = [q' \\ q'' = 10u- q]
def f(state, u):
    q, qdot = state
    return np.array([qdot, (u - k*q)/m])

data = []

for j in range(number_Trajectories):
    #u = 0.0    
    u= Neural_Network_One_Layer(n=10, input_dim=1, output_dim=1)
    #state = 0.1*np.random.randn(2)
    state = np.random.uniform(-1.5, 1.5, 2)   
    for i in range(number_time_points):
        q, qdot = state

        #u_val = 0.0 
        u_val = scale * u.ForwardPass(np.array([time_Grid[i]]))[0]
        qddot = (u_val - k*q)/m 

        # Spalten: id, t, q, qdot, u, qddot
        data.append([j, time_Grid[i], q, qdot, u_val, qddot])

        # RK4 zum nächsten Zustand
        k1 = f(state,           u_val)
        k2 = f(state + dt/2*k1, u_val)
        k3 = f(state + dt/2*k2, u_val)
        k4 = f(state + dt*k3,   u_val)
        state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

np.savetxt("spring_data.csv", data, delimiter=",", comments="", fmt="%.6f")