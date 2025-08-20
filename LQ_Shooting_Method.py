# LQ Steuerungsproblem
#======================================================================================
# min int_0^1 (u^2)/2 + x^2 dt, x' = (x/2) + u, x(0) =1
#--------------------------------------------------------------------------------------
# p'=-2x-(p/2) x'= (x/2)+u u=-p x(0) =1, p(1) = 0
# x = (e^3 e^(-3/2 t) + 2e^(3/2 t))/(2+e^3) 
# p = (2e^3 e^(-3/2)t -2 e^(3/2) t)/(2+e^3) 
#--------------------------------------------------------------------------------------

import math
import numpy as np
import algorithms


# Definition der Funktion zur Lösung unseres Randwertproblems
#------------------------------------------------------------------------------
def ODE_right_side(t,x):
   A = np.array(
    [
        [0.5 , -1],
        [-2, -0.5]
    ]
    )
   return np.dot(A,x) 

def Sensitivity_Matrix(x):
    gradient = np.array(
        [
            [
                0,-1
            ],
            [
                -3*x[0],0
            ]
        ]
    )

def Error_Term(ODE_right_side, initial_value):  
    solution_function = algorithms.solvingODE(ODE_right_side,initial_value)
    N = solution_function.shape[0]
    deviation = np.array(
        [initial_value[0] -4, initial_value[1] - 5*(solution_function[N-1][0]-1)]
    )
    return deviation

def Derivative_Error_Term(Sensitivity_Matrix, solution_function):
    Sensitivity = algorithms.solving_Matrix_ODE(Sensitivity_Matrix,solution_function)
    dimensions = Sensitivity.shape
    n = 2

    objective = Sensitivity[dimensions[0]-1][0:n][0:n]
    A = np.array(
        [
            [
                1,0
            ],
            [
                0,0
            ]
        ]
    )
    B = np.array(
        [
            [
                0,0
            ],
            [
                -5,1
            ]
        ]
    )
    objective = A + np.dot(B,objective)
    return objective
    
    
u = 0 #-2*(1-math.exp(3))/(2+math.exp(3))

#----------------------------------------------------------------------------
initial_value = np.array([1 ,u])

updated_initial_value = algorithms.iteration(ODE_right_side,Sensitivity_Matrix,initial_value,Error_Term,Derivative_Error_Term)

states = algorithms.solvingODE(ODE_right_side,updated_initial_value)

