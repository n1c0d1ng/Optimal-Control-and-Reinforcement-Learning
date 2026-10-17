# Implementierung einfacher Neuronal Networks
import numpy as np

# z_1 = W_1 * [t \\ x] + b_1 
# a_1 = tanh(z_1)
# V   = W_N a_1 + b_N
class Neural_Network_One_Layer:
    def __init__(self, n=5, input_dim=2, output_dim=1):
        self.n = n
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.parameter_dim = n * input_dim + n + output_dim * n + output_dim
        
        # Parameter initialisieren als (n,) statt (n,1) Vektoren
        self.W1 = np.random.randn(n, input_dim)     # (n, Input)
        self.b1 = np.random.randn(n)                # (n,)
        self.WN = np.random.randn(output_dim, n)    # (Output, n)
        self.bN = np.random.randn(output_dim)       # (Output,)

    def ForwardPass(self, Input_Vector):
        input_vec = Input_Vector                    # (Input,)
        z1 = self.W1 @ input_vec + self.b1          # (n,)
        a1 = np.tanh(z1)                            # (n,)
        V = self.WN @ a1 + self.bN                  # (1,)
        if self.output_dim == 1:
            return V[0], a1, z1                     # V (1,) V[0] skalar
        else:
            return V, a1, z1

    
    # flatten(): Geht Zeilenweise vor: [[1,2],[3,4]] -> [1,2,3,4]
    def Backpropagation(self, Input_Vector):
        V, a1, z1 = self.ForwardPass(Input_Vector)
        delta_1 = self.WN.flatten() * (1 - a1**2)   # (n,)
        delta_1 = delta_1.reshape(self.n, 1)        # (n,1)
        dVdWN = a1                                  # (n,)
        dVdbN = np.array([[1]])                     # (1,1)
        input_vec = Input_Vector.reshape(-1,1)      # (Input_Dim,1)
        dVdW1 = delta_1 @ input_vec.T               # (n,1) @ (1,Input_Dim) = (n,Input_Dim)
        dVdb1 = delta_1                             # (n,1)
        grad = np.concatenate([
            (dVdW1.T).flatten(),                    # (Input_Dim * n,)
            dVdb1.flatten(),                        # (n,)
            dVdWN.flatten(),                        # (n,)
            dVdbN.flatten()                         # (1,)
            ])                                      # (Input_Dim * n + 2n+1,)
        return grad
    
    def get_parameters(self):
        # Rückgabe als Vektor der Parameter
        params = np.concatenate([
            (self.W1.T).flatten(),
            self.b1.flatten(),
            self.WN.flatten(),
            self.bN.flatten()
        ])
        return params
    
    def set_parameters(self, params):
        cols = []
        index = 0

        for _ in range(self.input_dim):
            col = params[index:index+self.n]
            index += self.n
            cols.append(col)
        self.W1 = np.column_stack(cols)

        # b1
        self.b1 = params[index:index+self.n]
        index += self.n

        # W_N:
        WN_size = self.output_dim * self.n
        self.WN = params[index:index+WN_size].reshape(self.output_dim, self.n)
        index += WN_size

        # b_N:
        self.bN = params[index:index+self.output_dim]

    # Ableitung nach dem Input: 
    # V_x = delta^1 * W^1 
    # delta^1 = W^N * [1 - (a_1)^2] 
    def sensitivities(self, Input_Vector):
        a1 = self.ForwardPass(Input_Vector)[1]
        delta_1 = self.WN.flatten() * (1 - a1**2)   # (n,)
        grad_x = delta_1 @ self.W1                    # (input_dim,)
        return grad_x

# x'=f(x,u) = W^N tanh(W^1 x + b^1) + b^N 
# u =(W^1, b^1, W^N , b^N)

class Neural_ODE_MLP:
    def __init__(self, layers =10 ,dim_state=1, method='adjoint'):
        # h: layers n: dim_state
        self.layers = layers
        self.dim_state = dim_state

        # 2hn+h+n
        self.parameter_dim = 2*layers*dim_state+dim_state+layers
    
        # Parameter initialisieren als (n,) statt (n,1) Vektoren ???
        self.W1 = np.random.randn(layers, dim_state)    # (h, n)
        self.b1 = np.random.randn(layers)               # (h,)
        self.WN = np.random.randn(dim_state, layers)    # (n, h)
        self.bN = np.random.randn(dim_state)            # (n,)
        self.method = method

    # Einlesen der Daten: 
    def read_data(self, path):
        raw = np.loadtxt(path, delimiter=",", skiprows=0)
        M = (raw.shape[1] - 1) // self.dim_state       # Anzahl Trajektorien
        N = raw.shape[0]
        n = self.dim_state

        self.time_grid = raw[:, 0]                     # (N,)
        self.data = np.zeros((N, n, M))

        for j in range(M):
            for k in range(n):
                self.data[:, k, j] = raw[:, 1 + j * n + k]

        self.x0 = self.data[0]                         # (n, M)
        self.number_trajectories = M
        self.time_steps = N
        return self.time_grid, self.data

    
    # Forward Computation
    def forward(self):
        # M Trajektorien, N Zeitpunkte, Zustandsdimension n
        M = self.number_trajectories
        N = self.time_steps
        n = self.dim_state

        # X_hist[i, :, j] = x^j(t_i) speichert zum Zeitpunkt i den Wert der Trajektorie j
        self.X_hist = np.zeros((N, n, M))
        self.A_hist = np.zeros((N, self.layers, M))
        self.alpha_hist = np.zeros((N,self.layers,M))
        self.X_hist[0, :, :] = self.x0.reshape(n, M)

        for i in range(N - 1):
            dt = self.time_grid[i + 1] - self.time_grid[i]
            x = self.X_hist[i]                       

            # f = W^N tanh(W^1 x + b^1) + b^N
            # self.b1 ist (h,) self.b1[:,None] ist (h,1)
            z = self.W1 @ x + self.b1[:,None] @ np.ones((1,M))          
            a = np.tanh(z)      
            self.A_hist[i] = a       
            self.alpha_hist[i] = np.ones((self.layers,1)) @ np.ones((1,M)) - self.A_hist[i] * self.A_hist[i]         
            f = self.WN @ a + self.bN[:, None] @ np.ones((1,M))          
            self.X_hist[i + 1] = x + dt * f        
        z_last = self.W1 @ self.X_hist[N-1] + self.b1[:, None] @ np.ones((1, M))
        self.A_hist[N-1, :, :] = np.tanh(z_last)  
        return self.X_hist


    # Backward Integration
    def backward(self):
        M = self.number_trajectories
        N = self.time_steps
        n = self.dim_state

        self.P_hist = np.zeros((N, n, M))
        
        # Von Start:N-1 bis Ziel 0 (exklusiv, also 1) mit Schrittweite -1 (also Rückwärts)
        # für N-1 ist P_(N-1) = 0 als Randbedingung p(t_f) = 0
        for i in range(N-1, 0 ,-1):
            dt = self.time_grid[i] - self.time_grid[i-1]
            p = self.P_hist[i]

            # Warum bilde ich eigentlich alpha immer neu und speichere es nicht im forward pass ?
            #alpha = np.ones((self.layers,1)) @ np.ones((1,M)) - self.A_hist[i] * self.A_hist[i]
            res = 2*(self.X_hist[i] - self.data[i])
            self.P_hist[i-1]=  p + dt * (res + self.W1.T @ (self.alpha_hist[i] * (self.WN.T @ p)))
        return self.P_hist
    
    # Sensitivity ODE:
    #-----------------------------------------------------------------------------------------
    def Solve_SensitivityODE(self):
        M = self.number_trajectories
        N = self.time_steps
        n = self.dim_state
        h = self.layers

        # Für S_W^1 = [S^(1,1) S^(1,2) ... S^(h,n)] -> also h*n komponenten
        # (S_W^1^(j,k))_i = dx_i/dW_jk^1 wobei i die n Dimensionen von x sind 
        # Wir sammeln je Trajektorie eine Sammlung von S_W^1  -> M Komponenten
        # Nach b haben wir nur jeweils n oder h komponenten 

        self.S_W1_hist = np.zeros((N,n,h*n,M))
        self.S_b1_hist = np.zeros((N,n,h,M))
        self.S_WN_hist = np.zeros((N,n,n*h,M))
        self.S_bN_hist = np.zeros((N,n,n,M))

        for j in range(M):
            for i in range(N-1):
                dt = self.time_grid[i+1] - self.time_grid[i]
                a_i     = self.A_hist[i,:,j]                                # (n,)
                x_i     = self.X_hist[i,:,j]                                # (n,)
                alpha_i = self.alpha_hist[i,:,j]                            # (h,)
                f_x     = self.WN @ np.diag(alpha_i) @ self.W1              # (n,n)

                # Berechnung der U Terme:
                U_W1 = np.kron(self.WN @ np.diag(alpha_i), x_i[None,:])     # (n,h*n)
                U_b1 = self.WN @ np.diag(alpha_i)                           # (n,h)
                U_WN = np.kron(np.eye(n), a_i[None,:])                      # (n,n*h)
                U_bN = np.eye(n)                                            # (n,n)

                # Vorwärtsintegration von S
                self.S_W1_hist[i+1,:,:,j] = self.S_W1_hist[i,:,:,j] + dt * (f_x @ self.S_W1_hist[i,:,:,j] + U_W1)   
                self.S_b1_hist[i+1,:,:,j] = self.S_b1_hist[i,:,:,j] + dt * (f_x @ self.S_b1_hist[i,:,:,j] + U_b1)
                self.S_WN_hist[i+1,:,:,j] = self.S_WN_hist[i,:,:,j] + dt * (f_x @ self.S_WN_hist[i,:,:,j] + U_WN)
                self.S_bN_hist[i+1,:,:,j] = self.S_bN_hist[i,:,:,j] + dt * (f_x @ self.S_bN_hist[i,:,:,j] + U_bN)

    # Gradient update
    def gradient_descent(self):
        M = self.number_trajectories
        N = self.time_steps
        n = self.dim_state
        h = self.layers

        # self.layers = h
        gradJ_W1 = np.zeros((h,n))
        gradJ_b1 = np.zeros(h)
        gradJ_WN = np.zeros((n,h))
        gradJ_bN = np.zeros(n)

        # Gradient Adjoint Methode
        #------------------------------------------------------------------------------------
        if self.method == 'adjoint':
            for i in range(0,N-1):
                dt = self.time_grid[i+1] - self.time_grid[i]
                alpha = self.alpha_hist[i]
                delta = self.WN.T @ self.P_hist[i]
                gradJ_W1 += dt * (alpha*delta)  @ self.X_hist[i].T
                gradJ_b1 += dt * (alpha*delta)  @ np.ones(M)
                gradJ_WN += dt * self.P_hist[i] @ self.A_hist[i].T
                gradJ_bN += dt * self.P_hist[i] @ np.ones(M)

        # Gradient Sensitivity Methode
        #------------------------------------------------------------------------------------
        elif self.method == 'sensitivity':
            self.Solve_SensitivityODE()
            for j in range(M):
                for i in range(N-1):
                    res = 2*(self.X_hist[i,:,j] - self.data[i,:,j])
                    dt = self.time_grid[i+1] - self.time_grid[i]
                    gW1 = res @ self.S_W1_hist[i,:,:,j]
                    gb1 = res @ self.S_b1_hist[i,:,:,j]
                    gWN = res @ self.S_WN_hist[i,:,:,j]
                    gbN = res @ self.S_bN_hist[i,:,:,j]

                    # Summieren der Beiträge pro Zeitintervall
                    gradJ_W1 += dt * gW1.reshape(h, n)
                    gradJ_b1 += dt * gb1
                    gradJ_WN += dt * gWN.reshape(n, h)
                    gradJ_bN += dt * gbN
        return gradJ_W1, gradJ_b1, gradJ_WN, gradJ_bN

    # Update:
    def update(self, gradJ_W1, gradJ_b1, gradJ_WN, gradJ_bN, learning_rate = 0.01):
        self.W1 -= learning_rate * gradJ_W1
        self.b1 -= learning_rate * gradJ_b1
        self.WN -= learning_rate * gradJ_WN
        self.bN -= learning_rate * gradJ_bN

    def loss(self):
        M = self.number_trajectories
        N = self.time_steps
        loss = 0.0
        for i in range(N - 1):
            dt = self.time_grid[i + 1] - self.time_grid[i]
            diff = self.X_hist[i] - self.data[i]      # (n, M)
            loss += dt * np.sum(diff**2)
        return loss

    # Hier Schrittweitenbestimmung
    #-------------------------------------------------------------------------------
    def train_step(self, learning_rate_init=1e-3, beta=0.5, growth=1.1, c=1e-4, max_iter=20):
        # 1. Forward und Backward
        self.forward()
        loss_before = self.loss()
        self.backward()
        grads = self.gradient_descent()

        # 2. Armijo-Konstante: c * grad^2 
        grad_norm_sq = sum(np.sum(g**2) for g in grads)

        # 3. Parameter sichern
        W1_old = self.W1.copy()
        b1_old = self.b1.copy()
        WN_old = self.WN.copy()
        bN_old = self.bN.copy()

        # 4. Backtracking
        lr = learning_rate_init
        for _ in range(max_iter):
            self.update(*grads, learning_rate=lr)
            self.forward()
            loss_after = self.loss()

            # Armijo-Bedingung
            if loss_after <= loss_before - c * lr * grad_norm_sq:
                return loss_after, lr * growth

            # Zurücksetzen und halbieren
            self.W1 = W1_old.copy()
            self.b1 = b1_old.copy()
            self.WN = WN_old.copy()
            self.bN = bN_old.copy()
            lr *= beta

        return loss_before, lr

class LNN_trjectory:
    def __init__(self,layers =10, dim_state =1):
        self.h = layers
        self.dim_state = dim_state
        self.W1 = 0.1*np.random.randn(layers, 2) 
        self.b1 = np.zeros(layers)
        self.WN = 0.1*np.abs(np.random.randn(1, layers)) 
        self.bN = (-5)*np.ones(1)


    def read_data(self,path):
        raw = np.loadtxt(path, delimiter=",", skiprows=0)
        self.time_grid  = raw[:self.N, 1]
        self.M          = int(raw[:, 0].max()) + 1
        self.N          = raw.shape[0] // self.M

        self.X = np.zeros((self.N,2,self.M))
        self.Z = np.zeros((self.N,self.h,self.M))
        self.A = np.zeros((self.N,self.h,self.M))
        self.alpha = np.zeros((self.N,self.h,self.M))
        self.beta = np.zeros((self.N,self.h,self.M))
        self.gamma = np.zeros((self.N,self.h,self.M))
        self.C = np.zeros((self.N,1, self.M))
        self.H = np.zeros((self.N,1, self.M))
        self.dLdq = np.zeros((self.N,1, self.M))

        # Spalten: traj_id, t, q, qdot, u, qddot
        self.Q_start        = raw[:, 2]                             # (K,)
        self.Qdot_start     = np.zeros(self.M)                      # (M,)
        self.U              = raw[:, 4].reshape(self.M, self.N).T

        # Startwerte Speichern
        for k in range(self.M):
            self.Q_start[k]     = raw[k*self.N,2]
            self.Qdot_start[k]  = raw[k*self.N,3]

        # vstack -> [ [Werte Q] ; [Werte Qdot] ]
        self.X[0] = np.vstack([self.Q_start, self.Qdot_start])  # (2,M)

    def forward_pass(self):
        for i in range(self.N-1):
            self.Z[i] = self.W1 @ self.X[i] + self.b1[:,None] @ np.ones((1,self.M))
            self.A[i] = np.log(1+ np.exp(self.Z[i]))
            self.alpha[i] = np.exp(self.Z[i])/(1+np.exp(self.Z[i]))
            self.beta[i] = self.alpha[i]*(1-self.alpha[i])
            self.gamma[i] = self.beta[i]*(1-2*self.alpha[i])
            self.C[i],self.H[i], self.dLdq[i] = self.L_derivates(self.alpha[i],self.beta[i],self.gamma[i])

            # Euler Schritt für X
            dt = self.time_grid[i+1] - self.time_grid[i]
            self.X[i+1,0] = self.X[i,0] + dt*(self.X[i,1])
            self.X[i+1,1] = self.X[i,1] + dt*((self.dLdq[i, 0] - self.C[i, 0] * self.X[i,1] + self.U[i]) / self.H[i, 0])


    def L_derivates(self,alpha,beta,gamma):
        W1_1 = self.W1[:, 0][:,None]                       
        W1_2 = self.W1[:, 1][:,None]

        C = self.WN @ (beta* (W1_1 @ np.ones((1,self.M))) * (W1_2 @ np.ones((1,self.M))))
        # Lq = W^N (alpha * W1^1)
        dLdq = self.WN @ (alpha * (W1_1 @ np.ones((1,self.M))))
        # H = W^N (beta * W1^2 * W1^2)
        H = self.WN @ (beta * (W1_2 @ np.ones((1,self.M))) * (W1_2 @ np.ones((1,self.M))))
        return C ,H,dLdq

    def gradient_descent(self,X):
        return 0

    def train_step(self,X):
        return 0
    

class LNN_Regression:
    def __init__(self, layers =10 , activation_method = 'tanh' ,dim_state=1):
        # h: layers n: dim_state
        self.h = layers
        self.dim_state = dim_state
        self.activation = activation_method

        if self.activation == 'tanh':
            self.W1 = 0.1*np.random.randn(layers, 2) 
            self.b1 = (-1)*np.ones(layers)
            self.WN = 10.0 * np.abs(np.random.randn(1, layers)) + 10.0
            self.bN = np.zeros(1)
        elif self.activation == 'softplus':
            self.W1 = 0.1*np.random.randn(layers, 2) 
            self.b1 = np.zeros(layers)
            self.WN = 0.1*np.abs(np.random.randn(1, layers)) 
            self.bN = (-5)*np.ones(1)


    def read_data(self, path):
        raw = np.loadtxt(path, delimiter=",", skiprows=0)
        # Spalten: traj_id, t, q, qdot, u, qddot
        self.Q      = raw[:, 2]                         # (K,)
        self.Qdot   = raw[:, 3]                         # (K,)
        self.U      = raw[:, 4]                         # (K,)
        self.Qddot  = raw[:, 5]                         # (K,)
        # vstack -> [ [Werte Q] ; [Werte Qdot] ]
        self.X      = np.vstack([self.Q, self.Qdot])    # (2,K)
        self.M      = int(raw[:, 0].max()) + 1
        self.N      = raw.shape[0] // self.M
        # Hier K = M * N, also Anzahl der Datenpunkte
        self.K      = len(self.Q)
        return self.X, self.U, self.Qddot

    # Skalare Auswertung für einen konkreten Punkt
    def forward(self, q, qdot):
        X = np.array([q, qdot])                             # (2,)
        Z = self.W1 @ X + self.b1                           # (h,)
        if self.activation == 'tanh':
            A     = np.tanh(Z)                              # (h,)
            alpha = np.ones(self.h) - A*A                   # (h,)
            beta  = -2*A*alpha
            gamma = 2*alpha*(3*A*A - 1)
        elif self.activation == 'softplus':
            A = np.log(1+np.exp(Z))
            alpha = np.exp(Z)/(1+np.exp(Z))
            beta = alpha*(1 - alpha)
            gamma = beta*(1-2*alpha)
        else:
            raise ValueError(f"Unbekannte Aktivierung: {self.activation}")
        W1_1 = self.W1[:, 0]                            # (h,)
        W1_2 = self.W1[:, 1]

        dLdq    = self.WN @ (alpha * W1_1)              # (1,)
        dLdqdot = self.WN @ (alpha * W1_2)
        C       = self.WN @ (beta  * W1_1 * W1_2)
        H       = self.WN @ (beta  * W1_2 * W1_2)
        return dLdq, dLdqdot, C, H

    # Hier Berechnung der Hilfsgrößen a, ALPHA, BETA, GAMMA:
    def forward_all(self, X):
        # X = [X_1 X_2 ... X_N] und X_i ist 2x1 aber X ist (2,K=M*N)
        # self.b1 ist (h,) self.b1[:,None] ist (h,1)
        Z = self.W1 @ X + self.b1[:, None]@np.ones((1, self.K))         # (h,2) * (2, K) = (h, K)

        if self.activation == 'tanh':
            A     = np.tanh(Z)                                          # (h, K)
            alpha = np.ones((self.h, self.K)) - A*A                     # (h, K)
            beta  = -2*A*alpha                                          # (h, K)
            gamma = 2*alpha*(3*A*A - np.ones((self.h, self.K)))         # (h, K)
        elif self.activation == 'softplus':
            A = np.log(1+np.exp(Z))
            alpha = np.exp(Z)/(1+np.exp(Z))
            beta = alpha*(1 - alpha)
            gamma = beta*(1-2*alpha)
        else:
            raise ValueError(f"Unbekannte Aktivierung: {self.activation}")
        return A, alpha, beta, gamma   

        # Frage warum gilt hier Dimension K ? 
        # X: (2, K) -> (h, K)

    # Da die Punkte unabhängid sind genügt hier C als (1,K) abzuspeichern 
    # da dort alle N Zeitpunkte zu allen M Trajektorien drin sind.
    #-------------------------------------------------------------------------------------------
    def L_derivatives(self, X):

        A, alpha, beta, gamma = self.forward_all(X)
        # self.W1[:, 0] (h,) -> self.W1[:, 0][:, None] (h, 1)
        W1_1 = self.W1[:, 0][:, None]                               #   (h, 1)
        W1_2 = self.W1[:, 1][:, None]                               #   (h, 1)

        # dL/dq = W^{N} * ALPHA * (W^{1})^(1)
        dLdq    = self.WN @ (alpha * (W1_1@np.ones((1, self.K))))  #    (1,h)*(h,K) = (1,K)

        # dL/dq' = W^{N} * ALPHA * (W^{1})^(2)
        dLdqdot = self.WN @ (alpha * (W1_2@np.ones((1, self.K))))   #   (1,h)*(h,K) = (1,K)

        # C = W^{N} * BETA * [ (W^{1})^(1) * (W^{1})^(2)]               (1,h)*(h,K) = (1,K)
        C = self.WN @ (beta * (W1_1@np.ones((1, self.K))) * (W1_2@np.ones((1, self.K)))) 

        # H =  W^{N} * BETA * [ (W^{1})^(2) * (W^{1})^(2)]              (1,h)*(h,K) = (1,K)
        #-----------------------------------------------------------------------------------------
        H = self.WN @ (beta * (W1_2@np.ones((1, self.K))) * (W1_2@np.ones((1, self.K))))
        return dLdq, dLdqdot, C, H

    def gradient_descent(self):
        K       = self.K                                        # K = M*N
        X       = self.X                                        # (2, K)
        Q       = self.Q[None, :]                               # (1, K)
        Qd      = self.Qdot[None, :]                            # (1, K)
        U       = self.U[None, :]                               # (1, K)
        Qdd     = self.Qddot[None, :]                           # (1, K)

        # (1_M W^N)^T wobei M hier K also alle Zeitpunkte ist.
        WN_Batch    = (np.ones((self.K,1))@(self.WN)).T         # (h, K
        W1_1        = self.W1[:, 0][:, None]                    # (h, 1)
        W1_2        = self.W1[:, 1][:, None]                    # (h, 1)

        A, alpha, beta, gamma = self.forward_all(X)
        dLdq, dLdqdot , C, H = self.L_derivatives(X)

        # dH/dtheta (h, K): Wir verwenden K statt M da wir alle N Zeitpunkte betrachten und K= M*N
        #-----------------------------------------------------------------------------------------
        # dH/d(W^{1})^(1) = (1_M @ WN)^T * (gamma * (1_h @ Q) * (W1_2 @ 1_M^T)^2)
        dHdW1_1 = WN_Batch * (
            gamma * (np.ones((self.h,1)) @ Q) * (W1_2@np.ones((1,self.K)))**2)              # (h, K)
        # dH/d(W^{1})^(2) = 
        #       (1_M @ WN)^T * (W1_2 @ 1_M^T) * ((W1_2 @ 1_M^T) * gamma * (1_h @ Q') + 2*beta)
        dHdW1_2 = WN_Batch * (W1_2 @ np.ones((1,self.K))) * (
            (W1_2 @ np.ones((1,self.K))) * gamma * (np.ones((self.h,1)) @ Qd) + 2*beta)     # (h, K)
        # dH/db^{1} = (1_M @ WN)^T * gamma * (W1_2 @ 1_M^T)^2
        dHdb1   = WN_Batch * gamma * (W1_2 @ np.ones((1,self.K)))**2
        # dH/d(W^{N}) = beta * (W1_2 @ 1_M^T)^2
        dHdWN   = beta * (W1_2 @ np.ones((1,self.K)))**2

        # dN/dtheta (h, K): dN/dtheta = d/dtheta dL/dq - dC/dtheta * Q'
        #----------------------------------------------------------------------------------------
        # dC/d(W^{1})^(1) = (1_M @ WN)^T * (W1_2 @ 1_M^T) * (gamma * (1_h @ Q) * (W1_1 @ 1_M^T) + beta)
        dCdW1_1 = WN_Batch * (W1_2 @ np.ones((1,self.K))) * (
            gamma * (np.ones((self.h,1)) @ Q) * (W1_1 @ np.ones((1,self.K))) + beta)      # (h, K)
        # dC/d(W^{1})^(2) = (1_M @ WN)^T * (W1_1 @ 1_M^T) * ((W1_2 @ 1_M^T) * gamma * (1_h @ Q) + beta)
        dCdW1_2 = WN_Batch * (W1_1 @ np.ones((1,self.K))) * (
            gamma * (np.ones((self.h,1)) @ Qd) * (W1_2 @ np.ones((1,self.K))) + beta)      # (h, K)
        # dC/db^{1} = (1_M @ WN)^T * (W1_1 @ 1_M^T) * (W1_2 @ 1_M^T) * gamma
        dCdb1   = WN_Batch * (W1_1 @ np.ones((1,self.K))) * (W1_2 @ np.ones((1,self.K))) * gamma
        # dC/d(W^{N}) = (W1_1 @ 1_M^T) * (W1_2 @ 1_M^T) * beta
        dCdWN   = beta * (W1_1 @ np.ones((1,self.K))) * (W1_2 @ np.ones((1,self.K)))
        # d/(W^{1})^(1) dL/dq = (WN^T @ 1_M^T) * (beta * (1_h @ Q) * (W1_1 @ 1_M^T) + alpha)
        dLdqdW1_1 = WN_Batch * (
            beta*(np.ones((self.h,1)) @ Q)*(W1_1 @ np.ones((1,self.K)))+alpha)
        # d/(W^{1})^(2) dL/dq = (WN^T @ 1_M^T) * beta * (1_h @ Q') * (W1_2 @ 1_M^T)
        dLdqdW1_2 = WN_Batch * (
            beta * (np.ones((self.h,1)) @ Qd) * (W1_1 @ np.ones((1,self.K))))
        # d/(b^{1}) dL/dq = (WN^T @ 1_M^T) * beta * (W1_1 @ 1_M^T)
        dLdqdb1   = WN_Batch * beta * (W1_1 @ np.ones((1,self.K)))
        # d/(W^{N}) dL/dq = alpha * (W1_1 @ 1_M^T)
        dLdqdWN   = alpha * (W1_1 @ np.ones((1,self.K)))

        # Zusammensetzen der Ableitungen
        #---------------------------------------------------------------------------------------------
        dNdW1_1 = dLdqdW1_1 - dCdW1_1 * Qd                              # (h, K)
        dNdW1_2 = dLdqdW1_2 - dCdW1_2 * Qd                              # (h, K)
        dNdb1   = dLdqdb1   - dCdb1   * Qd                              # (h, K)
        dNdWN   = dLdqdWN   - dCdWN   * Qd                              # (h, K)

        # P_i^j = 2*(Q'' - Qdd) / H wobei Q''= H^{-1}(dL/dq - C Q' + U) = N/
        # Hier is die einzige stelle in der U eingeht
        #---------------------------------------------------------------------------------------------
        N = dLdq - C*Qd + U                                             # (1, K)
        qddot_pred = N / H                                              # (1, K)
        P = 2*(qddot_pred - Qdd) / H                                    # (1, K)

        # Gradientenblöcke aufstellen:
        # nabla J= 1/NM * sum(P_i^j * (dN/dtheta - dH/dtheta * Q'') )
        #--------------------------------------------------------------------------------------------- 
        gW11 = np.sum(P * (dNdW1_1 - dHdW1_1*qddot_pred), axis=1) / K   # (h,)
        gW12 = np.sum(P * (dNdW1_2 - dHdW1_2*qddot_pred), axis=1) / K   # (h,)
        gb1  = np.sum(P * (dNdb1   - dHdb1  *qddot_pred), axis=1) / K   # (h,)
        gWN  = np.sum(P * (dNdWN   - dHdWN  *qddot_pred), axis=1) / K   # (h,)

        grad_W1 = np.column_stack([gW11, gW12])                         # (h, 2)
        grad_b1 = gb1                                                   # (h,)
        grad_WN = gWN.reshape(1, self.h)                                # (1, h)
        return grad_W1, grad_b1, grad_WN

    # Hier wird der Gradientenschritt durchgezogen auf die Parameter
    def train_step(self, learning_rate=0.01):
        gW1, gb1, gWN = self.gradient_descent()
        self.W1 -= learning_rate * gW1                                  # (h, 2)
        self.b1 -= learning_rate * gb1                                  # (h,)
        self.WN -= learning_rate * gWN                                  # (1, h)    

    # J(theta) = 1/NM * sum_i^N sum_j^M (Q'' - Qdd)^2   K = 1/MN
    def loss(self):
        dLdq, dLdqdot , C, H = self.L_derivatives(self.X)
        Qd  = self.Qdot[None, :]
        U   = self.U[None, :]
        Qdd = self.Qddot[None, :]
        # Hier geht die Steuerung ein:
        N = dLdq - C*Qd + U
        qddot_pred = N / H
        error = qddot_pred - Qdd
        return float(np.mean(error**2))

# z_1 = W_1 * [t \\ x] + b_1 
# a_1 = tanh(z_1)
# z_2 = W_2 * a_1 + b_2
# a_2 = tanh(z_2)
# z_N = W_N a_2 + b_N
# Under Contruction
class Neural_Network_Two_Layer:
    def __init__(self, n=5, input_dim=2, output_dim=1):
            self.n = n
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.parameter_dim = n * input_dim + n + output_dim * n + output_dim
            
            # Parameter initialisieren als (n,) statt (n,1) Vektoren
            self.W1 = np.random.randn(n, input_dim)     # (n, 2)
            self.b1 = np.random.randn(n)                # (n,)
            self.WN = np.random.randn(output_dim, n)    # (1, n)
            self.bN = np.random.randn(output_dim)       # (1,)

    def ForwardPass(self, t, X):
        input_vec = np.array([t, X])                # (2,)
        z1 = self.W1 @ input_vec + self.b1          # (n,)
        a1 = np.tanh(z1)                            # (n,)
        V = self.WN @ a1 + self.bN                  # (1,)
        if self.output_dim == 1:
            return V[0], a1, z1                     # V (1,) V[0] skalar
        else:
            return V, a1, z1

    
    # flatten(): Geht Zeilenweise vor: [[1,2],[3,4]] -> [1,2,3,4]
    def Backpropagation(self, t, X):
        V, a1, z1 = self.ForwardPass(t, X)
        delta_1 = self.WN.flatten() * (1 - a1**2)   # (n,)
        delta_1 = delta_1.reshape(self.n, 1)        # (n,1)
        dVdWN = a1                                  # (n,)
        dVdbN = np.array([[1]])                     # (1,1)
        input_vec = np.vstack([t, X])               # (2,1)
        dVdW1 = delta_1 @ input_vec.T               # (n,1) @ (1,2) = (n,2)
        dVdb1 = delta_1                             # (n,1)
        grad = np.concatenate([
            (dVdW1.T).flatten(),                    # (2n,)
            dVdb1.flatten(),                        # (n,)
            dVdWN.flatten(),                        # (n,)
            dVdbN.flatten()                         # (1,)
            ])                                      # (4n+1,)
        return grad
    
    def get_parameters(self):
        """Gibt alle Parameter als flachen Vektor zurück"""
        params = np.concatenate([
            (self.W1.T).flatten(),
            self.b1.flatten(),
            self.WN.flatten(),
            self.bN.flatten()
        ])
        return params
    
    def set_parameters(self, params):
        index = 0
        
        # W_1:
        Column_1 = params[index:index+self.n]
        index += self.n
        Column_2 = params[index:index+self.n]
        index += self.n
        self.W1 = np.column_stack([Column_1, Column_2])  # (n,2)
        
        # b_1: n Elemente
        self.b1 = params[index:index+self.n]    # (n,)
        index += self.n
        
        # W_N: output_dim * n Elemente
        wn_size = self.output_dim * self.n
        self.WN = params[index:index+wn_size].reshape(self.output_dim, self.n) # (output_dim, n)
        index += wn_size
        
        # bN: output_dim Elemente
        self.bN = params[index:index+self.output_dim] # (output_dim,)