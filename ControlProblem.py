# Diese Datei dient zur Definition simpler Steuerungsprobleme
#-------------------------------------------------------------------------------------------------
import numpy as np

class ControlProblem:
# min \int_0^1 (x + u^2) dt + x(1)^2
# mit der Zustandsgleichung x' = x + u + 1, x(0) = 0
    def __init__(self, N=100):
        self.N = N
        self.T = 1
        self.h = self.T / N
        self.t = np.linspace(0, self.T, N + 1)
        self.u = np.zeros(N + 1)  # Initiale Steuerung
        self.x = np.zeros(N + 1)  # Initialer Zustand

    # x' = x + u + 1, x(0) = 0
    def forward_integration(self, u):
        x  = np.zeros(self.N + 1)
        for i in range(self.N):
            x[i + 1] = x[i] + self.h * (x[i] + u[i] + 1)
        return x
    
    # p' = -(1+p) p(1) =  -> p_k = p_{k+1} - (- h*(1+p_{k+1}))
    def backward_integration(self):
        p = np.zeros(self.N + 1)
        p[self.N] = self.x[self.N]  # Endwert
        for i in range(self.N):
        #for i in range(self.N - 1, -1, -1):
            p[self.N -i -1] = p[self.N - i] + self.h * (1 + p[self.N - i])
            #p[i] = p[i + 1] + self.h * (1 + p[i + 1])
        return p
    
    # Gradient H_u = 2u + p
    def compute_gradient(self, p):
        return -2 * self.u - p
    
    # Bedingung: J(u_new) <= J(u^k) - c * beta^m J'(u^k)^2 
    def armijo_step(self, d,x ,c=1e-4, beta=0.5, iter_max=10):
        u = self.u.copy()
        for m in range(iter_max):
            u_new = u + (beta**m)*d
            J_old = self.compute_cost(x,u)
            J_new = self.compute_cost(self.forward_integration(u_new), u_new)
            # Prüfe Armijo Bedingung
            if J_new <= J_old - c*(beta**m)*np.linalg.norm(d)**2:
                break
        return beta**m
    
    # J(u) = h * sum_i=0^´N-1 (x_i + u_i^2) + (x_N)^2
    def compute_cost(self,x, u):
        return self.h* np.sum(x + u**2) + x[self.N]**2

    def gradient_descent(self, max_iter=100, tol=1e-6):
        for k in range(max_iter):
            x = self.forward_integration(self.u)
            p = self.backward_integration()
            d = self.compute_gradient(p)

            # Schrittweite bestimmen
            alpha = self.armijo_step(d, x)
            if np.linalg.norm(self.u + alpha * d - self.u) < tol:
                break
            # Steuerung aktualisieren
            self.u = self.u + alpha * d
            self.x = self.forward_integration(self.u)

    # p' = -(1+p), p(1) = 2x(1) -> p(t) = -1 + A exp(-t) -> A = 1 + 2x(1) exp(1)
    # p = (6e^2 -4e/ e^2 + 1) exp(-t) -1
    def p_analytic(self):
        return ((6 * np.exp(2) - 4 * np.exp(1)) / (np.exp(2) + 1)) * np.exp(-self.t) - 1

    # H_u = 0 -> u = -p/2 -> u = (-1/2) (A exp(-t) -1)
    # u = 1/2 - (3e^2 -2e/ e^2 + 1) exp(-t)
    def u_analytic(self):
        return 0.5 - ((3*np.exp(2) - 2*np.exp(1)) / (np.exp(2) + 1)) * np.exp(-self.t)

    # x' = x + u + 1, x(0) = 0 -> x(t) = B exp(t) + A/4 exp(-t) - 3/2
    # x = [(3+2e)exp(t) + (3e^2 -2e)exp(-t)]/2(e^2 +1) - 3/2
    def x_analytic(self):
        return ((3 + 2 * np.exp(1)) * np.exp(self.t) + (3 * np.exp(2) - 2 * np.exp(1)) * np.exp(-self.t)) \
            / (2 * (np.exp(2) + 1)) - 1.5

    def get_next_state(self, x, u):
        return x + self.h * (x + u + 1)

class SimpleControlProblem:
# min \int_0^1 u^2 dt + (x(1)-1)^2
# mit der Zustandsgleichung x' = u, x(0) = 0
# H = u^2 + p*u -> H_u = 2u + p , H_x = 0 -> p' = 0

    def __init__(self, N=100):
        self.N = N
        self.T = 1
        self.h = self.T / N
        self.t = np.linspace(0, self.T, N + 1)
        self.u = np.zeros(N + 1)  # Initiale Steuerung
        self.x = np.zeros(N + 1)  # Initialer Zustand

    def forward_integration(self, u):
        x  = np.zeros(self.N + 1)
        for i in range(self.N):
            x[i + 1] = x[i] + self.h * (u[i])
        return x

    # p'=-H_x = 0 -> p = const -> p_k = p_{k+1}
    def backward_integration(self):
        p = np.zeros(self.N + 1)
        p[self.N] = 2 * self.x[self.N]  # Endwert
        for i in range(self.N):
            p[self.N -i -1] = p[self.N - i]  # da p' = 0
        return p
    
    # Gradient H_u = 2u + p 
    def compute_gradient(self, p):
        return -2 * self.u - p
    
    def compute_running_cost(self,x, u):
        return self.h* np.sum(u**2) + x[self.N]**2
    
    def compute_terminal_cost(self,x, u):
        return x[self.N]**2
    
    # x'=u=-p/2 -> x(t) = -(p/2)*t -> x(1) = -(p(1)/2)-> x(t)=t/2
    def x_analytic(self):
        return 0.5*self.t

    # p'=-H_x =0 -> p(t)=p(1)=2(x(1)-1)=2(-p(1)/2 -1)-> p(t)=-1
    def p_analytic(self):
        return -1.0*np.ones(self.N + 1)
    
    # H_u = 2u+p = 0 -> u = -p/2 -> u = 1/2
    def u_analytic(self):
        return 0.5*np.ones(self.N + 1)