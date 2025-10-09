import numpy as np
from scipy.integrate import solve_ivp

def gradient_descent_optimal_control():
    # Diskretisierung
    N = 1000  # Anzahl Zeitpunkte
    t_span = (0, T)
    t_eval = np.linspace(0, T, N)
    h = T / (N-1)
    
    # Initial guess für Steuerung (kann auch u_f sein)
    u_trajectory = np.zeros((2, N))
    
    # Gradientenabstiegsparameter
    max_iter = 100
    alpha = 0.01  # Schrittweite
    tol = 1e-6
    
    for iter in range(max_iter):
        # 1. Vorwärtsintegration: Zustandstrajektorie berechnen
        x_trajectory = np.zeros((4, N))
        x_trajectory[:, 0] = x0
        
        for i in range(N-1):
            # Numerische Integration der nichtlinearen Dynamik
            k1 = f(x_trajectory[:, i], u_trajectory[:, i])
            k2 = f(x_trajectory[:, i] + 0.5*h*k1, 0.5*(u_trajectory[:, i] + u_trajectory[:, i+1]))
            k3 = f(x_trajectory[:, i] + 0.5*h*k2, 0.5*(u_trajectory[:, i] + u_trajectory[:, i+1]))
            k4 = f(x_trajectory[:, i] + h*k3, u_trajectory[:, i+1])
            
            x_trajectory[:, i+1] = x_trajectory[:, i] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        
        # 2. Rückwärtsintegration: Adjungierte berechnen
        lambda_trajectory = np.zeros((4, N))
        lambda_trajectory[:, -1] = 0  # Randbedingung
        
        for i in range(N-2, -1, -1):
            # Adjungierte Gleichung rückwärts integrieren
            x = x_trajectory[:, i]
            u = u_trajectory[:, i]
            lambda_val = lambda_trajectory[:, i+1]
            
            # Hamilton-Gradient bezüglich x
            dHdx = compute_dHdx(x, u, lambda_val)
            
            # Rückwärtsintegration (Euler)
            lambda_trajectory[:, i] = lambda_val - h * dHdx
        
        # 3. Gradient berechnen
        gradient = np.zeros_like(u_trajectory)
        for i in range(N):
            x = x_trajectory[:, i]
            u = u_trajectory[:, i]
            lambda_val = lambda_trajectory[:, i]
            
            gradient[:, i] = 2 * R @ (u - u_f) + M_inv(x).T @ lambda_val[2:]  # nur Geschwindigkeitsteil
        
        # 4. Steuerung updaten
        u_new = u_trajectory - alpha * gradient
        
        # 5. Konvergenz check
        if np.linalg.norm(gradient) < tol:
            print(f"Konvergenz nach {iter} Iterationen")
            break
        
        u_trajectory = u_new
    
    return u_trajectory, x_trajectory

def compute_dHdx(x, u, lambda_val):
    """Berechnet ∂H/∂x für die Roboter-Dynamik"""
    q = x[:2]
    dq = x[2:]
    
    # Ableitungen berechnen (kann numerisch oder analytisch)
    dLdx = np.zeros(4)
    dLdx[:2] = 2 * Q @ (q - q_f)  # Ableitung nach q
    # dLdx[2:] = 0  # Ableitung nach dq ist 0
    
    # Ableitung der Dynamik f nach x
    dfdx = compute_dfdx(q, dq, u)
    
    dHdx = dLdx + dfdx.T @ lambda_val
    return dHdx

def compute_dfdx(q, dq, u):
    """Berechnet Jacobi-Matrix ∂f/∂x analytisch oder numerisch"""
    # Hier müssten die analytischen Ableitungen der Roboter-Dynamik implementiert werden
    # Vereinfacht: numerische Differentiation
    eps = 1e-8
    n = 4
    dfdx = np.zeros((n, n))
    x = np.concatenate([q, dq])
    
    for i in range(n):
        x_perturbed = x.copy()
        x_perturbed[i] += eps
        f_plus = f(x_perturbed, u)
        
        x_perturbed[i] = x[i] - eps
        f_minus = f(x_perturbed, u)
        
        dfdx[:, i] = (f_plus - f_minus) / (2*eps)
    
    return dfdx