import numpy as np

def vec(X):
    """Vektorisiere Matrix X (Spaltenweise)."""
    return X.reshape(-1, order='F')

def unvec(v, n):
    """Forme Vektor v zurück in n×n Matrix (Spaltenweise)."""
    return v.reshape((n, n), order='F')

def solve_riccati_newton(A, B, R, Q, tol=1e-8, max_iter=50):
    """
    Löse Q - K P K + K A + A^T K = 0 für K mit P = B R^{-1} B^T
    mittels vektorisierter Newton-Iteration.
    """
    n = A.shape[0]
    I = np.eye(n)
    # Konstant: P = B R^{-1} B^T
    P = B @ np.linalg.solve(R, B.T)
    
    # Startwert K0 (z.B. Nullmatrix)
    K = np.zeros((n, n))
    x = vec(K)
    
    # Vorberechne q und L
    q = vec(Q)
    L = np.kron(I, A.T) + np.kron(A.T, I)
    
    for _ in range(max_iter):
        # (2a) X = K P
        X = K @ P
        
        # (2b) Residuum f(x) = q - vec(KPK) + (A^T⊗I)x + (I⊗A^T)x
        # vec(KPK) = vec(X K) = (I⊗X) x
        vec_KPK = np.kron(I, X) @ x
        f = q - vec_KPK + L @ x
        
        # (3) Jacobian J = L - [I⊗X + X^T⊗I]
        Dg = np.kron(I, X) + np.kron(X, I)
        Df = L - Dg
        
        # (4) Newton-Schritt: Δx = solve(J, -f)
        dx = np.linalg.solve(Df, -f)
        x_new = x + dx
        
        # Abbruch
        if np.linalg.norm(dx) / np.linalg.norm(x_new) < tol:
            x = x_new
            break
        
        x = x_new
        K = unvec(x, n)
    
    # Endgültige Lösung
    return unvec(x, n)

# Beispielaufruf
if __name__ == "__main__":
    n, m = 4, 2
    A = np.random.randn(n, n)
    B = np.random.randn(n, m)
    R = np.eye(m)
    Q = np.eye(n)
    K = solve_riccati_newton(A, B, R, Q)
    print("Lösung K:\n", K)
