# Wir behandeln folgendes Steuerungsproblem
# 
#-------------------------------------------------------------------------------------------------
import numpy as np

#--------------------------------------------------------------------------------------------------
def initialize(N=100):
    T = 1 
    h = T/N
    t = np.linspace(0,T,N+1) #genau N+1 Elemente
    u = np.zeros(N+1)
    x = np.zeros(N+1)
    return t,h,x,u

# hier ist state euqation x' = x+u+1 x(0) = 0
#--------------------------------------------------------------------------------------------------
def forward_integration(x,u,h):
    N = len(x) -1
    for i in range(N):
        x[i+1] = x[i] + h * (x[i] + u[i] +1)

    return x


# hier ist adjungierte Gleichung p' = -(1+p) p(1) = 0
#--------------------------------------------------------------------------------------------------
def backward_integration(p,x,u,h):
    N = len(p) -1
    for i in range(N-1,-1,-1): 
        # Startwert N-1, Endwert -1 ausschließlich also 0, step -1 also absteigend
        p[i]= p[i+1] + h * (1+p[i+1]) 
    return p 


# Gradient H_u = 2u + p
#--------------------------------------------------------------------------------------------------
def compute_gradient(u,p):
    d = -2*u -p 
    return d


# Bedingung: J(u_new) <= J(u^k) - c * beta^m J'(u^k)^2 
#--------------------------------------------------------------------------------------------------
def armijo_step(u,d,x,p,h,c=1e-4,beta=0.5, iter_max = 10):
    # Anweisung
    for m in range(iter_max):
        u_new = u + (beta**m)*d 
        J_old = compute_cost(x,u,h)
        J_new = compute_cost(forward_integration(x,u_new,h),u_new,h)

        # Prüfe Armijo Bedingung 
        if J_new <= J_old - c*(beta**m)*np.linalg.norm(d)**2:
            break
    return beta**m


# J(u) = h * sum_i=0^´N-1 (x_i + u_i^2)
#--------------------------------------------------------------------------------------------------
def compute_cost(x,u,h):
    return h* np.sum(x + u**2)

# Gradientenverfahren
#--------------------------------------------------------------------------------------------------
def gradient_descent(N, max_iter=100, tol=1e-6):
    t, h, u, x = initialize(N)
    for k in range(max_iter):
        # Vorwärtsintegration
        x = forward_integration(x, u, h)
        # Rückwärtsintegration
        p = np.zeros(N+1)
        p = backward_integration(p, x, u, h)
        # Gradienten berechnen
        d = compute_gradient(u, p)
        # Schrittweite bestimmen
        alpha = armijo_step(u, d, x, p, h)
        # Steuerung aktualisieren
        u_new = u + alpha *d
        # Abbruchbedingung prüfen
        if np.linalg.norm(u_new - u) < tol:
            break
        u = u_new
    return u, x, t