# Testumfeld der logistischen ODE
#-------------------------------------------------
import numpy as np
import csv

def Logistics_Map(x0,t,K=2,a=0.5,t0 =0):
    return (x0*K)/((K-x0)*np.exp(-a*K*(t-t0))+x0)

starting_points = np.array([0.1, 0.3, 0.5, 0.7 ,0.8,  0.9 , 1.0 , 1.1 , 1.2, 1.3])
time_grid = np.linspace(0, 5, 100)

# Matrix: Zeilen = Zeitpunkte, Spalten = Trajektorien
data = np.zeros((len(time_grid), len(starting_points)))
for j in range(len(starting_points)):
    data[:, j] = Logistics_Map(starting_points[j], time_grid)

np.savetxt("logistic_data.csv", 
           np.column_stack([time_grid, data]),
           delimiter=",", comments="", fmt="%.6f")