import numpy as np
from scipy.linalg import solve_continuous_lyapunov as lyap

def vec(X):
    """Vectorize matrix X in column-major (Fortran) order."""
    return X.reshape(-1, order='F')

def unvec(v, n):
    """Unvectorize vector v into n x n matrix (column-major)."""
    return v.reshape((n, n), order='F')

def solve_continuous_riccati_newton(A, B, R, Q, tol=1e-8, max_iter=50):
    """
    Solve the continuous-time algebraic Riccati equation (CARE)
    using the vectorized Newton method:
    
        A^T P + P A - P B R^{-1} B^T P + Q = 0
    
    Inputs:
    - A, B: system matrices (A: n×n, B: n×m)
    - R: control weighting matrix (m×m)
    - Q: state weighting matrix (n×n)
    - tol: convergence tolerance
    - max_iter: maximum number of Newton iterations
    
    Returns:
    - P: solution matrix (n×n)
    """
    n = A.shape[0]
    I = np.eye(n)
    
    # Initial guess P0 via Lyapunov: A^T P0 + P0 A + Q = 0
    P = lyap(A.T, -Q)
    v = vec(P)
    
    # Precompute constant kron parts
    q = vec(Q)
    L = np.kron(I, A.T) + np.kron(A.T, I)  # for linear terms
    
    for k in range(max_iter):
        # Compute X = P B R^{-1} B^T
        B_Rinv_BT = B @ np.linalg.solve(R, B.T)
        X = P @ B_Rinv_BT
        
        # Compute vec(P B R^{-1} B^T P) = vec(X P) = (I⊗X) v
        vec_PBP = np.kron(I, X) @ v
        
        # Compute f(v) = L v - vec_PBP + q
        f = L @ v - vec_PBP + q
        
        # Build Frechet derivative Dg = (I⊗X + X^T ⊗ I)
        Dg = np.kron(I, X) + np.kron(X.T, I)
        
        # Build full Jacobian Df(v) = L - Dg
        J = L - Dg
        
        # Solve J dv = -f
        dv = np.linalg.solve(J, -f)
        
        # Update
        v_new = v + dv
        P_new = unvec(v_new, n)
        
        if np.linalg.norm(dv) / np.linalg.norm(v_new) < tol:
            P = P_new
            break
        
        v = v_new
        P = P_new
    
    return P

# Example usage:
if __name__ == "__main__":
    n, m = 4, 2
    np.random.seed(0)
    A = np.random.randn(n, n)
    B = np.random.randn(n, m)
    R = np.eye(m)
    Q = np.eye(n)
    
    P = solve_continuous_riccati_newton(A, B, R, Q)
    print("Solution P:\n", P)
