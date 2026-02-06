# Stochastic Evironment
# min_u \int_0^1 (X_t + u_t^2) dt + X_1^2
# dX_t = (X_t + u_t +1) dt + dW_t, X_0 = 
import numpy as np

class SOCP:
    def __init__(self, N=100):
        self.N = N
        self.T = 1
        self.sigma = 0.5
        self.h = self.T / N
        self.t = np.linspace(0, self.T, N + 1)

    def get_next_state(self, X, u):
        # Simulation dX_t = (X_t + u_t +1) dt + sigma dW_t 
        Z = np.random.normal(0, 1)  
        return X + self.h * (X + u + 1) + self.sigma * Z * np.sqrt(self.h)
    
    def compute_running_cost(self, X, u):
        return self.h *  (np.sum(X) + np.sum(u**2)) 
    
    def compute_terminal_cost(self, X):
        return X[-1]**2
    
    # u = 1/2 - (3e^2 -2e/ e^2 + 1) exp(-t)
    def u_analytic(self):
        return 0.5 - ((3*np.exp(2) - 2*np.exp(1)) / (np.exp(2) + 1)) * np.exp(-self.t)

    # x = [(3+2e)exp(t) + (3e^2 -2e)exp(-t)]/2(e^2 +1) - 3/2
    def x_analytic(self):
        return ((3 + 2 * np.exp(1)) * np.exp(self.t) + (3 * np.exp(2) - 2 * np.exp(1)) * np.exp(-self.t)) \
            / (2 * (np.exp(2) + 1)) - 1.5
    
