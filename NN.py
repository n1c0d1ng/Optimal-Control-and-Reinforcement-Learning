# Implementierung einfacher Neural Networks
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


# Hier explizit die Neural ODE als Funktionsapproximator:
class NeuralODE:
    def __init__(self, layers=10, integration_steps =10, input_dim = 2 , output_dim =1):
        self.h = layers
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.integration_steps = integration_steps
        self.step_size = 1/self.integration_steps
        self.augmented_dimension = self.h - input_dim
        self.W = np.random.randn(self.h,self.h)                                     # (h, h)
        self.b = np.random.randn(self.h)                                            # (h,)
        self.WN = np.random.randn(output_dim, self.h)                               # (output_dim, h)
        self.bN = np.random.randn(output_dim)                                       # (output_dim,)
        self.a_hist = np.zeros(
            (self.integration_steps, self.h))                                       # (integration_steps, h)
        self.alpha_hist = np.zeros(
            (self.integration_steps, self.h))                                       # (integration_steps, h)
        self.p_hist = np.zeros(
            (self.integration_steps, self.h, output_dim))                           # (integration_steps, h, output_dim)
        self.parameter_dim = (self.h)**2 + self.h + self.output_dim * self.h + self.output_dim

    def compute(self, input_vector):
        # Augmentieren: a0 = [x_1, x_2, 0, 0, ..., 0] ->  (h,)
        a0 = np.pad(input_vector, (0, self.augmented_dimension), 'constant')

        # Integration a' = tanh(W a + b) - a
        a = a0[:,None]       
        self.a_hist[0] = a0                                                         # (h,1)
        for i in range(self.integration_steps):
            # RK4 a'(s) = h(a(s)) = tanh(W*a(s)+b) - a(s) -> autonom
            #------------------------------------------------
            h = self.step_size
            k1 = np.tanh(self.W @ a + self.b[:,None]) - a
            k2 = np.tanh(self.W @ (a+h/2 * k1) + self.b[:,None]) - (a+ h/2 *k1)
            k3 = np.tanh(self.W @ (a+h/2 * k2) + self.b[:,None]) - (a+ h/2 *k2)
            k4 = np.tanh(self.W @ (a+h * k3) + self.b[:,None]) - (a+ h*k3)
            ode_step = 1/6 * (k1 + 2*k2 + 2*k3 + k4)
            a = a + h * ode_step
            if i < self.integration_steps - 1:
                self.a_hist[i+1] = a[:,0]  

        # Ausgabe: (output_dim,1) = (output_dim,h) @ (h,1) + (output_dim,1)
        self.a_final = a[:,0]                                                       # (h,)
        output = (self.WN @ a + self.bN)[:,0]
        if self.output_dim == 1:
            return output  # Rückgabe als Skalar
        return output

    def solve_adjoint(self, input_vector):
        self.compute(input_vector)
        p = self.WN.T.copy()

        for i in range(self.integration_steps - 1, -1, -1):
            a_i = self.a_hist[i]
            if i < self.integration_steps -1:
                a_ip1 = self.a_hist[i+1]
            else:
                a_ip1 = self.a_final
            a_mid = 0.5 * (a_i + a_ip1)

            # a linear interpolieren und damit auch alpha
            #-------------------------------------------------
            alpha = 1 - np.tanh(self.W @ a_i + self.b)**2
            alpha_mid = 1 - np.tanh(self.W @ a_mid + self.b)**2
            alpha_next = 1 - np.tanh(self.W @ a_ip1 + self.b)**2
            self.alpha_hist[i] = alpha
            # RK4: p'(s) = - dh/da(s) * p(s) und p(1)= (W^N)^T
            #-------------------------------------------------
            # k1 = f(t,x)
            # k2 = f(t + h/2, x + h/2 k1)
            # k3 = f(t + h/2, x + h/2 k2)
            # k4 = f(t + h,   x + h k3)
            # x_n+1 = x_n + h/6 k1 + h/3 k2 + h/3 k3 + h/6 k4
            M = -(self.W.T @ np.diag(alpha) - np.eye(self.h))                       # (h,h)
            M_mid = -(self.W.T @ np.diag(alpha_mid) - np.eye(self.h)) 
            M_next = -(self.W.T @ np.diag(alpha_next) - np.eye(self.h)) 
            h = -self.step_size
            k1 = M_next @ p
            k2 = M_mid @ (p + 0.5*h*k1)
            k3 = M_mid @ (p + 0.5*h*k2)
            k4 = M @ (p + h*k3)
            p = p + h/6 * (k1 + 2*k2 + 2*k3 + k4)
            self.p_hist[i] = p

        if self.output_dim == 1:
            return p[:self.input_dim,0]                                             # (input_dim,)
        else:
            return p[:self.input_dim]                                               # (input_dim, output_dim)


    def computeDerivative(self, input_vector):
        d = self.output_dim
        dgdW  = np.zeros((d, self.h, self.h))                                       # (d, h, h)
        dgdb  = np.zeros((d, self.h))                                               # (d, h)
        dgdWN = np.zeros((d, d, self.h))                                            # (d, d, h)
        dgbN  = np.zeros((d, d))                                                    # (d, d)
        p0 = self.solve_adjoint(input_vector)                                       # (input_dim, d)

        for i in range(self.integration_steps):
            alpha_i = self.alpha_hist[i]                                            # (h,)
            a_i     = self.a_hist[i]                                                # (h,)
            p_i     = self.p_hist[i]                                                # (h, d) = (h, 1)
            for m in range(d):
                # dg/dW = int_0^1 (alpha_i * p_i[:, m]) @ a_i^T ds
                dgdW[m] += self.step_size * (alpha_i * p_i[:, m])[:, None] * a_i[None, :]
                # dg/db = int_0^1 (alpha_i * p_i[:, m]) ds
                dgdb[m] += self.step_size * (alpha_i * p_i[:, m])

        for m in range(d):
            # dg/WN = a(1)^T
            dgdWN[m, m, :] = self.a_final
        dgbN = np.eye(d)

        # dg/dbN = 1
        if self.output_dim == 1:
            return (dgdW[0], dgdb[0], dgdWN[0], dgbN[0], p0)
        else:
            return (dgdW, dgdb, dgdWN, dgbN, p0)

    # Schnitstelle für Policy Gradient Methode:
    #---------------------------------------------------------------
    def ForwardPass(self, Input_Vector):
        return self.compute(Input_Vector)

    def Backpropagation(self, Input_Vector):
        dgdW, dgdb, dgdWN, dgbN, _ = self.computeDerivative(Input_Vector)
        grad = np.concatenate([
            dgdW.flatten(),
            dgdb.flatten(),
            dgdWN.flatten(),
            dgbN.flatten()
        ])
        return grad

    def sensitivities(self, Input_Vector):
        _, _, _, _, p0 = self.computeDerivative(Input_Vector)
        return p0
    
    def get_parameters(self):
        params = np.concatenate([
            self.W.flatten(),
            self.b,
            self.WN.flatten(),
            self.bN
        ])
        return params

    def set_parameters(self, params):
        index = 0
        W_size = self.h * self.h
        self.W = params[index:index + W_size].reshape(self.h, self.h)
        index += W_size

        b_size = self.h
        self.b = params[index:index + b_size]
        index += b_size

        WN_size = self.output_dim * self.h
        self.WN = params[index:index + WN_size].reshape(self.output_dim, self.h)
        index += WN_size

        bN_size = self.output_dim
        self.bN = params[index:index + bN_size]



# Hier Adjoint und Sensitivity Methode 
# für allgemeine Neural ODE um Dynamik zu lernen:
#----------------------------------------------------------------------------
class Neural_ODE_MLP:
    def __init__(self, layers =10 ,dim_state=1,dim_control =1 , method='adjoint'):
        # h: layers n: dim_state
        self.layers = layers
        self.dim_state = dim_state
        self.dim_control = dim_control

        # 2hn+h+n+(hm)
        self.parameter_dim = 2*layers*dim_state+dim_state+layers+layers*dim_control
        self.method = method

    # Einlesen der Daten: 
    def read_data(self, path):
        raw = np.loadtxt(path, delimiter=",", skiprows=0)                   # (N*M,2+n+m)
        # Spalten: traj_id, t, x_1 - x_n, u_1 - u_m
        #-----------------------------------------------------------------
        self.number_trajectories            = int(raw[:, 0].max()) + 1
        self.time_steps                     = raw.shape[0] // self.number_trajectories
        self.time_grid                      = raw[:self.time_steps, 1]      # (N,)
        self.X_data = np.zeros((
            self.time_steps, self.dim_state, self.number_trajectories))     # (N,n,M)
        self.U_data = np.zeros((
            self.time_steps, self.dim_control, self.number_trajectories))   # (N,m,M)

        for j in range(self.number_trajectories):
            for k in range(self.dim_state):
                self.X_data[:,k,j] = raw[
                    j*self.time_steps:(j+1)*self.time_steps,
                    2+k] 
            for k in range(self.dim_control):
                self.U_data[:,k,j] = raw[
                    j*self.time_steps:(j+1)*self.time_steps,
                    2+self.dim_state+k] 
                
        # Hier LeChun Regel: Paper GLeichung 16
        # Var(W) = 1/(fan-in * Var(Input))
        # #-------------------------------------------------------------------------------    
        self.W1x = np.random.randn(self.layers, self.dim_state)               # (h, n)
        self.W1u = np.random.randn(self.layers, self.dim_control)             # (h, m)
        self.b1 = np.random.randn(self.layers)                                # (h,)
        self.WN = np.random.randn(self.dim_state, self.layers)                # (n, h)
        self.bN = np.random.randn(self.dim_state)                             # (n,)
        self.x0 = self.X_data[0]                                              # (n,M)

    # Forward Computation
    def forward(self):
        # M Trajektorien, N Zeitpunkte, Zustandsdimension n
        M = self.number_trajectories
        N = self.time_steps
        n = self.dim_state
        m = self.dim_control
        h = self.layers

        # X_hist[i, :, j] = x^j(t_i) speichert zum Zeitpunkt i den Wert der Trajektorie j
        self.X_hist = np.zeros((N, n, M))
        self.A_hist = np.zeros((N, h, M))
        self.alpha_hist = np.zeros((N, h, M))
        self.X_hist[0, :, :] = self.x0

        for i in range(N - 1):
            dt = self.time_grid[i + 1] - self.time_grid[i]
            x = self.X_hist[i]     
            u = self.U_data[i]                  

            # f = W^N tanh(W^1 x + b^1) + b^N
            # self.b1 ist (h,) self.b1[:,None] ist (h,1)
            z = self.W1x @ x + self.W1u @ u + self.b1[:,None] @ np.ones((1,M))          
            a = np.tanh(z)      
            self.A_hist[i] = a       
            self.alpha_hist[i] = np.ones((h,1)) @ np.ones((1,M)) - self.A_hist[i]**2   
            f = self.WN @ a + self.bN[:, None] @ np.ones((1,M))          
            self.X_hist[i + 1] = x + dt * f        
        z_last = self.W1x @ self.X_hist[N-1] + self.W1u @ self.U_data[N-1] + self.b1[:, None] @ np.ones((1, M))
        self.A_hist[N-1, :, :] = np.tanh(z_last)  
        self.alpha_hist[N-1,:,:] = np.ones((h,1)) @ np.ones((1,M)) - self.A_hist[N-1]**2
        return self.X_hist


    # Backward Integration
    def backward(self):
        M = self.number_trajectories
        N = self.time_steps
        n = self.dim_state
        m = self.dim_control
        self.P_hist = np.zeros((N, n, M))
        
        # Von Start:N-1 bis Ziel 0 (exklusiv, also 1) mit Schrittweite -1 (also Rückwärts)
        # für N-1 ist P_(N-1) = 0 als Randbedingung p(t_f) = 0
        for i in range(N-1, 0 ,-1):
            dt = self.time_grid[i] - self.time_grid[i-1]
            p = self.P_hist[i]
            res = 2*(self.X_hist[i] - self.X_data[i])
            # Hier nur Euler: Mal auf RK gehen ?
            #-------------------------------------------------------------------------
            self.P_hist[i-1] =  p + dt * (
                res + self.W1x.T @ (self.alpha_hist[i] * (self.WN.T @ p)))
        return self.P_hist
    
    # Sensitivity ODE:
    #-----------------------------------------------------------------------------------------
    def Solve_SensitivityODE(self):
        M = self.number_trajectories
        N = self.time_steps
        n = self.dim_state
        m = self.dim_control
        h = self.layers

        # Für S_W^1 = [S^(1,1) S^(1,2) ... S^(h,n)] -> also h*n komponenten
        # (S_W^1^(j,k))_i = dx_i/dW_jk^1 wobei i die n Dimensionen von x sind 
        # Wir sammeln je Trajektorie eine Sammlung von S_W^1  -> M Komponenten
        # Nach b haben wir nur jeweils n oder h komponenten 

        self.S_w1x = np.zeros((N,n,h*n,M))
        self.S_w1u = np.zeros((N,n,h*m,M))
        self.S_b1 = np.zeros((N,n,h,M))
        self.S_wN = np.zeros((N,n,n*h,M))
        self.S_bN = np.zeros((N,n,n,M))

        for j in range(M):
            for i in range(N-1):
                dt = self.time_grid[i+1] - self.time_grid[i]
                a_i     = self.A_hist[i,:,j]                                # (n,)
                x_i     = self.X_hist[i,:,j]                                # (n,)
                u_i     = self.U_data[i,:,j]                                # (m,)
                alpha_i = self.alpha_hist[i,:,j]                            # (h,)
                f_x     = self.WN @ np.diag(alpha_i) @ self.W1x             # (n,n)

                # Berechnung der U Terme:
                U_w1x = np.kron(self.WN @ np.diag(alpha_i), x_i[None,:])    # (n,h*n)
                U_w1u = np.kron(self.WN @ np.diag(alpha_i), u_i[None,:])    # (n,h*m)
                U_b1 = self.WN @ np.diag(alpha_i)                           # (n,h)
                U_wN = np.kron(np.eye(n), a_i[None,:])                      # (n,n*h)
                U_bN = np.eye(n)                                            # (n,n)

                # Vorwärtsintegration von S
                # Parallel für jede Zustandsdimension und jeden Parameter von theta
                self.S_w1x[i+1,:,:,j] = self.S_w1x[i,:,:,j] + dt * (f_x @ self.S_w1x[i,:,:,j] + U_w1x)   
                self.S_w1u[i+1,:,:,j] = self.S_w1u[i,:,:,j] + dt * (f_x @ self.S_w1u[i,:,:,j] + U_w1u) 
                self.S_b1[i+1,:,:,j] = self.S_b1[i,:,:,j] + dt * (f_x @ self.S_b1[i,:,:,j] + U_b1)
                self.S_wN[i+1,:,:,j] = self.S_wN[i,:,:,j] + dt * (f_x @ self.S_wN[i,:,:,j] + U_wN)
                self.S_bN[i+1,:,:,j] = self.S_bN[i,:,:,j] + dt * (f_x @ self.S_bN[i,:,:,j] + U_bN)

    # Gradient update
    def compute_gradient(self):
        M = self.number_trajectories
        N = self.time_steps
        n = self.dim_state
        m = self.dim_control 
        h = self.layers

        gradJ_W1x = np.zeros((h,n))
        gradJ_W1u = np.zeros((h,m))
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
                gradJ_W1x += dt * (alpha*delta)  @ self.X_hist[i].T
                gradJ_W1u += dt * (alpha*delta)  @ self.U_data[i].T
                gradJ_b1 += dt * (alpha*delta)  @ np.ones(M)
                gradJ_WN += dt * self.P_hist[i] @ self.A_hist[i].T
                gradJ_bN += dt * self.P_hist[i] @ np.ones(M)

        # Gradient Sensitivity Methode
        #------------------------------------------------------------------------------------
        elif self.method == 'sensitivity':
            self.Solve_SensitivityODE()
            for j in range(M):
                for i in range(N-1):
                    res = 2*(self.X_hist[i,:,j] - self.X_data[i,:,j])
                    dt = self.time_grid[i+1] - self.time_grid[i]
                    gW1x = res @ self.S_w1x[i,:,:,j]
                    gW1u = res @ self.S_w1u[i,:,:,j]
                    gb1 = res @ self.S_b1[i,:,:,j]
                    gWN = res @ self.S_wN[i,:,:,j]
                    gbN = res @ self.S_bN[i,:,:,j]

                    # Summieren der Beiträge pro Zeitintervall
                    gradJ_W1x += dt * gW1x.reshape(h, n)
                    gradJ_W1u += dt * gW1u.reshape(h, m)
                    gradJ_b1 += dt * gb1
                    gradJ_WN += dt * gWN.reshape(n, h)
                    gradJ_bN += dt * gbN
        return gradJ_W1x, gradJ_W1u, gradJ_b1, gradJ_WN, gradJ_bN

    # Update:
    def update(
            self, 
            gradJ_W1x, 
            gradJ_W1u, 
            gradJ_b1, 
            gradJ_WN, 
            gradJ_bN, 
            learning_rate = 0.01):
        self.W1x -= learning_rate * gradJ_W1x
        self.W1u -= learning_rate * gradJ_W1u
        self.b1 -= learning_rate * gradJ_b1
        self.WN -= learning_rate * gradJ_WN
        self.bN -= learning_rate * gradJ_bN

    def loss(self):
        M = self.number_trajectories
        N = self.time_steps
        dt = self.time_grid[1] - self.time_grid[0]
        loss = 0.0
        for i in range(N):
            diff = self.X_hist[i] - self.X_data[i]                      # (n, M)
            loss += dt * np.sum(diff**2)
        return loss

    # Ein Gradientenabstiegsschritt
    #-------------------------------------------------------------------------------
    def train_step(
            self, 
            learning_rate_init=1e-3, 
            beta=0.5, 
            growth=1.1, 
            c=1e-4, 
            max_iter=20):
        
        # 1. Forward und Backward
        self.forward()
        loss_before = self.loss()
        self.backward()
        grads = self.compute_gradient()

        # 2. Armijo-Konstante: c * grad^2 
        grad_norm_sq = sum(np.sum(g**2) for g in grads)

        # 3. Parameter sichern
        W1x_old = self.W1x.copy()
        W1u_old = self.W1u.copy()
        b1_old = self.b1.copy()
        WN_old = self.WN.copy()
        bN_old = self.bN.copy()

        # 4. Backtracking
        #-------------------------------------------------------------
        lr = learning_rate_init
        for _ in range(max_iter):
            self.update(*grads, learning_rate=lr)
            self.forward()
            loss_after = self.loss()

            # Armijo-Bedingung
            if loss_after <= loss_before - c * lr * grad_norm_sq:
                return loss_after, lr * growth

            # Zurücksetzen und halbieren
            self.W1x = W1x_old.copy()
            self.W1u = W1u_old.copy()
            self.b1 = b1_old.copy()
            self.WN = WN_old.copy()
            self.bN = bN_old.copy()
            lr *= beta
        return loss_before, lr

    # Fertiges Netz auswerten:
    #---------------------------------------------------------------------------------
    def f_point(self, x, u):
        x = np.atleast_1d(x); u = np.atleast_1d(u)
        z = self.W1x @ x + self.W1u @ u + self.b1
        return self.WN @ np.tanh(z) + self.bN

    def dfdx(self, x, u):
        x = np.atleast_1d(x); u = np.atleast_1d(u)
        z = self.W1x @ x + self.W1u @ u + self.b1
        alpha = 1 - np.tanh(z)**2
        return self.WN @ np.diag(alpha) @ self.W1x

    def dfdu(self, x, u):
        x = np.atleast_1d(x); u = np.atleast_1d(u)
        z = self.W1x @ x + self.W1u @ u + self.b1
        alpha = 1 - np.tanh(z)**2
        return self.WN @ np.diag(alpha) @ self.W1u

class LNN_trjectory:
    def __init__(self,layers =10, dim_state =1):
        self.h = layers
        self.dim_state = dim_state
        self.W1 = 0.1*np.random.randn(layers, 2) 
        self.b1 = np.zeros(layers)
        self.WN = 0.1*np.abs(np.random.randn(1, layers)) 
        self.bN = (-5)*np.ones(1)
        self.epsilon = 0.1
        self.sub = 4


    def read_data(self,path):
        raw = np.loadtxt(path, delimiter=",", skiprows=0)
        self.M          = int(raw[:, 0].max()) + 1
        self.N          = raw.shape[0] // self.M
        self.time_grid  = raw[:self.N, 1]
        self.colNumber = 5                              
        self.reset_interval = int(self.N // self.colNumber)         # 500/5 = 100
        self.mask_hist = np.zeros((self.N,self.M))

        self.X = np.zeros((self.N,2,self.M))                        # (N,2,M)
        self.X_sub = np.zeros((self.N,self.sub,2,self.M))           # (N,sub=4,2,M)
        self.Z = np.zeros((self.N,self.h,self.M))                   # (N,h,M)
        self.A = np.zeros((self.N,self.h,self.M))                   # (N,h,M)
        self.alpha = np.zeros((self.N,self.h,self.M))               # (N,h,M)
        self.beta = np.zeros((self.N,self.h,self.M))                # (N,h,M)
        self.gamma = np.zeros((self.N,self.h,self.M))               # (N,h,M)
        self.C = np.zeros((self.N,1, self.M))                       # (N,1,M)
        self.H = np.zeros((self.N,1, self.M))                       # (N,1,M)
        self.dLdq = np.zeros((self.N,1, self.M))                    # (N,1,M)

        # Historien für die Rückwärtsintegration:
        self.dHdq       = np.zeros((self.N, 1, self.M))             # (N,1,M)
        self.dHdqdot    = np.zeros((self.N, 1, self.M))             #
        self.dCdq       = np.zeros((self.N, 1, self.M))             #
        self.dCdqdot    = np.zeros((self.N, 1, self.M))             # 
        self.dLdqq      = np.zeros((self.N, 1, self.M))             #

        # Spalten: traj_id, t, q, qdot, u, qddot
        self.Q_start        = np.zeros(self.M)                      # (M,)
        self.Qdot_start     = np.zeros(self.M)                      # (M,)
        self.Q_data         = np.zeros((self.N,self.M))             # (N,M)
        self.Qdot_data      = np.zeros((self.N,self.M))             # (N,M)
        self.U              = raw[:, 4].reshape(self.M, self.N).T

        # Startwerte Speichern
        for k in range(self.M):
            self.Q_start[k]     = raw[k*self.N,2]
            self.Qdot_start[k]  = raw[k*self.N,3]
            for i in range(self.N):
                self.Q_data[i,k]        = raw[k*self.N+i,2]
                self.Qdot_data[i,k]     = raw[k*self.N+i,3]

        # vstack -> [ [Werte Q] ; [Werte Qdot] ]
        self.X[0] = np.vstack([self.Q_start, self.Qdot_start])  # (2,M)

    def forward_pass(self):
        for i in range(self.N):
            self.Z[i] = self.W1 @ self.X[i] + self.b1[:,None] @ np.ones((1,self.M))
            self.A[i] = np.log(1+ np.exp(self.Z[i]))
            self.alpha[i] = np.exp(self.Z[i])/(1+np.exp(self.Z[i]))
            self.beta[i] = self.alpha[i]*(1-self.alpha[i])
            self.gamma[i] = self.beta[i]*(1-2*self.alpha[i])
            self.C[i], self.H[i], self.dLdq[i], self.dHdq[i], \
                self.dLdqq[i], self.dCdq[i], self.dHdqdot[i], \
                self.dCdqdot[i], self.mask_hist[i] = (
                    self.L_derivates(
                        self.alpha[i], 
                        self.beta[i], 
                        self.gamma[i]
                    )
                )
            
            # Euler Schritt für X außer am letzten Schritt:
            if i < self.N-1:
                dt = self.time_grid[i+1] - self.time_grid[i]
                step_size  = dt / self.sub
                q       = self.X[i, 0].copy()
                qdot    = self.X[i, 1].copy()
                u_i     = self.U[i]
                for k in range(self.sub):
                    q, qdot = self.RK4_step(q, qdot, u_i, step_size)
                    self.X_sub[i, k, 0] = q
                    self.X_sub[i, k, 1] = qdot
                self.X[i+1, 0] = q
                self.X[i+1, 1] = qdot

    # Skalare Auswertung für einen konkreten Punkt
    def forward(self, q, qdot):
        X = np.array([q, qdot])                             # (2,)
        Z = self.W1 @ X + self.b1                           # (h,)
        A = np.log(1+np.exp(Z))
        alpha = np.exp(Z)/(1+np.exp(Z))
        beta = alpha*(1 - alpha)
        gamma = beta*(1-2*alpha)
        W1_1 = self.W1[:, 0]                            # (h,)
        W1_2 = self.W1[:, 1]

        dLdq    = self.WN @ (alpha * W1_1)              # (1,)
        dLdqdot = self.WN @ (alpha * W1_2)
        C       = self.WN @ (beta  * W1_1 * W1_2)
        H       = np.maximum(self.WN @ (beta  * W1_2 * W1_2),self.epsilon)
        return dLdq, dLdqdot, C, H
    
    def lagrangian_term(self, q , qdot):
        X = np.vstack([q, qdot])                                    # (2, M)
        Z = self.W1 @ X + self.b1[:, None] @ np.ones((1,self.M))    # (h, M)
        # softplus:
        A     = np.log(1 + np.exp(Z))
        alpha = np.exp(Z) / (1 + np.exp(Z))
        beta  = alpha * (1 - alpha)
        W1_1  = self.W1[:, 0:1]
        W1_2  = self.W1[:, 1:2]
        dLdq  = self.WN @ (alpha * W1_1)
        C     = self.WN @ (beta * W1_1 * W1_2)
        H_raw = self.WN @ (beta * W1_2 * W1_2)
        H     = np.maximum(H_raw, self.epsilon)
        return dLdq[0], C[0], H[0]
    
    def f(self, q, qdot, u):
        Lq, C, H = self.lagrangian_term(q, qdot)
        qddot = (Lq - C*qdot + u) / H
        return qdot, qddot

    def RK4_step(self, q, qdot, u, step_size):
        k1q, k1qdot = self.f(q, qdot, u)
        k2q, k2qdot = self.f(q + 0.5*step_size*k1q, qdot + 0.5*step_size*k1qdot, u)
        k3q, k3qdot = self.f(q + 0.5*step_size*k2q, qdot + 0.5*step_size*k2qdot, u)
        k4q, k4qdot = self.f(q + step_size*k3q, qdot + step_size*k3qdot, u)
        q_new = q + step_size/6 * (k1q + 2*k2q + 2*k3q + k4q)
        qdot_new = qdot + step_size/6 * (k1qdot + 2*k2qdot + 2*k3qdot + k4qdot)
        return q_new, qdot_new
    
    def L_derivates(self,alpha,beta,gamma):
        W1_1 = self.W1[:, 0][:,None] @ np.ones((1,self.M))                      
        W1_2 = self.W1[:, 1][:,None] @ np.ones((1,self.M))

        C = self.WN @ (beta*W1_1*W1_2)
        # Lq = W^N (alpha * W1^1)
        dLdq = self.WN @ (alpha * W1_1)
        # H = W^N (beta * W1^2 * W1^2)
        H_raw = self.WN @ (beta * W1_2* W1_2)
        mask = H_raw > self.epsilon                # (1,M)
        H = np.maximum(H_raw, self.epsilon)

        # Ergänzen im Pseudo Code: Broadcast für jede spalte mit M 
        dHdq = mask * (self.WN @ ((gamma*W1_1*W1_2*W1_2)))
        dLdqq = self.WN @ ((beta*W1_1*W1_1))
        dCdq = self.WN @ ((gamma*W1_1*W1_1*W1_2))
        dHdqdot = mask * (self.WN @ ((gamma*W1_2*W1_2*W1_2)))
        dCdqdot = self.WN @ ((gamma*W1_2*W1_1*W1_2))

        return C ,H,dLdq , dHdq, dLdqq, dCdq, dHdqdot, dCdqdot, mask

    def compute_f_x(self,i):
        # X zum Zeitpunkt i auswerten: 
        qdot    = self.X[i,1]                                       # (M,)
        N       = self.dLdq[i,0] - self.C[i,0] * qdot + self.U[i]   # (M,)

        # Erste Komponente von (f_x)*P:
        # p_2 * ((1/H^2) * H_q (L_q - C q' + u) +1/H (L_qq - C_q *q'))
        fxP_1 = self.P[i,1]*( \
            -(1/(self.H[i,0]*self.H[i,0])) * self.dHdq[i,0] * N \
            + 1/self.H[i,0] *(self.dLdqq[i,0]- self.dCdq[i,0]*qdot))

        # Zweite Komponente von (f_x)*P:
        # p_1 + p_2 *(-1/H^2 * H_q' (L_q - C q' + u) + 1/H (C- C_q' q' - C))
        fxP_2 = self.P[i,0]+self.P[i,1]*(-1/(self.H[i,0]*self.H[i,0]) * self.dHdqdot[i,0]*N \
                                        + 1/self.H[i,0] * (- self.dCdqdot[i,0] * qdot) ) 
        return fxP_1, fxP_2

    def compute_f_x_single(self, i, p1, p2):
        # Auswertung von f_x * P an beliebigen Werten von P
        qdot    = self.X[i,1]                                       # (M,)
        N       = self.dLdq[i,0] - self.C[i,0] * qdot + self.U[i]   # (M,)
        H       = self.H[i,0]
        Hq      = self.dHdq[i,0]
        Lqq     = self.dLdqq[i,0]
        Cq      = self.dCdq[i,0]
        Hqdot   = self.dHdqdot[i,0]
        Cqdot   = self.dCdqdot[i,0]

        fxP_1 = p2 * ( -Hq / H**2 * N + (Lqq - Cq * qdot) / H)
        fxP_2 = p1 + p2 * (-Hqdot / H**2 * N - Cqdot * qdot / H)
        return fxP_1, fxP_2

    def backward_pass(self):
        self.P = np.zeros((self.N,2,self.M))
        for i in range(self.N-1,0,-1):
            dt  = self.time_grid[i] - self.time_grid[i-1]
            p1 = self.P[i, 0].copy()
            p2 = self.P[i, 1].copy()
            res_q       = 2 * (self.X[i, 0] - self.Q_data[i])   # Loss-Gradient
            res_qdot    = np.zeros_like(res_q)

            def rhs(p1, p2):
                fxP_1, fxP_2 = self.compute_f_x_single(i, p1, p2)
                return -fxP_1 - res_q, -fxP_2 - res_qdot

            k1q, k1qdot = rhs(p1,             p2)
            k2q, k2qdot = rhs(p1 - 0.5*dt*k1q, p2 - 0.5*dt*k1qdot)
            k3q, k3qdot = rhs(p1 - 0.5*dt*k2q, p2 - 0.5*dt*k2qdot)
            k4q, k4qdot = rhs(p1 - dt*k3q,     p2 - dt*k3qdot)
            self.P[i-1, 0] = np.clip(p1 - dt/6 * (k1q + 2*k2q + 2*k3q + k4q),-10,10)
            self.P[i-1, 1] = np.clip(p2 - dt/6 * (k1qdot + 2*k2qdot + 2*k3qdot + k4qdot),-10,10)


    def compute_gradient(self):
        # Broadcast der Gewichte:
        WN_T = (self.WN.T) @ np.ones((1,self.M))                                        # (h,M)
        W1_1 = self.W1[:, 0:1] @ np.ones((1,self.M))                                    # (h,M)
        W1_2 = self.W1[:, 1:2] @ np.ones((1,self.M))            


        # Anlegen der Arrays für den Gradienten:
        grad_W1 = np.zeros((self.h, 2))                                                 # (h,2)
        grad_b1 = np.zeros(self.h)                                                      # (h,)
        grad_WN = np.zeros(self.h)                                                      # (h,)

        dt = self.time_grid[1] - self.time_grid[0]
        for i in range(self.N):
            q_i         = np.ones((self.h,1)) @ self.X[i, 0][None, :]                   # (h,M)
            qdot_i      = self.X[i, 1][None, :]                                         # (1,M)
            qdot_i_B    = np.ones((self.h,1)) @ qdot_i                                  # (h,M)
            H_i         = self.H[i, 0][None, :]                                         # (1,M)
            N_i         = self.dLdq[i, 0] - self.C[i, 0]*qdot_i + self.U[i]             # (M,)
            # Berechnung der Preciction von q'' nach dem aktuellen Modell:
            # q'' = H^-1 (L_q - C q' + u)
            qddot_i = N_i / H_i                                                         # (1,M)
            qddot_i_B = np.ones((self.h,1)) @ qddot_i                                   # (h,M)
            P2      = self.P[i, 1]                                                      # (1,M)
            mask = (np.ones((self.h,1)) @ self.mask_hist[i][None,:])                    # (h,M)

            # Berechnung dH/dtheta
            dHdW1_1 = mask * WN_T * W1_2 * W1_2 * self.gamma[i] * q_i                           # (h,M)
            dHdW1_2 = mask * WN_T * W1_2 * (self.gamma[i] * W1_2 * qdot_i_B + 2*self.beta[i])   # (h,M)
            dHdb1   = mask * self.gamma[i]* W1_2 * W1_2                                         # (h,M)
            dHdWN   = mask * self.beta[i] * W1_2 * W1_2                                         # (h,M)

            # Berechnung dC/dtheta
            dCdW1_1 = WN_T * W1_2 * (self.gamma[i] * W1_1 * q_i + self.beta[i])
            dCdW1_2 = WN_T * W1_1 * (self.gamma[i] * W1_2 * qdot_i_B + self.beta[i])
            dCdb1   = WN_T * self.gamma[i] * W1_1 * W1_2
            dCdWN   = self.beta[i] * W1_1 * W1_2

            # Berechnung dL/dtheta
            dLqdW1_1    = WN_T * (self.beta[i] * W1_1 * q_i + self.alpha[i])
            dLqdW1_2    = WN_T * (self.beta[i] * W1_1 * qdot_i_B)
            dLqdb1      = WN_T * (self.beta[i] * W1_1)
            dLqdWN      = self.alpha[i] * W1_1

            # Gradient berechnen: J_theta = sum_i 1/H *(-H_theta q'' + d/dtheta N) * p_2
            # N = L_q - C q' + u 
            grad_W1[:,0] += dt * 1/(np.ones((self.h,1)) @ self.H[i]) * (-dHdW1_1 * qddot_i_B + dLqdW1_1 - dCdW1_1 * qdot_i_B) @ P2
            grad_W1[:,1] += dt * 1/(np.ones((self.h,1)) @ self.H[i]) * (-dHdW1_2 * qddot_i_B + dLqdW1_2 - dCdW1_2 * qdot_i_B) @ P2
            grad_b1[:]   += dt * 1/(np.ones((self.h,1)) @ self.H[i]) * (-dHdb1 * qddot_i_B + dLqdb1 - dCdb1 * qdot_i_B) @ P2
            grad_WN[:]   += dt * 1/(np.ones((self.h,1)) @ self.H[i]) * (-dHdWN * qddot_i_B + dLqdWN - dCdWN * qdot_i_B) @ P2

        return grad_W1/(self.M * self.N), grad_b1/(self.M * self.N) , grad_WN.reshape(1,self.h)/(self.M * self.N)
        
    def train_step(self,learning_rate= 0.01):
        self.forward_pass()
        self.backward_pass()
        grad_W1, grad_b1, grad_WN = self.compute_gradient()
    
        self.W1 -= learning_rate * grad_W1          # (h,2)
        self.b1 -= learning_rate * grad_b1          # (h,)
        self.WN -= learning_rate * grad_WN          # (1,h)

    def loss(self):
        # sum_i (q-q_data)^2 
        error = self.X[:,0,:] - self.Q_data                 # (N,M)
        #dt = self.time_grid[1] - self.time_grid[0]
        return float(np.mean((error**2)))

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

class LNN_structured:
    def __init__(self,layers =10, dim_state =1):
        self.h = layers
        self.dim_state = dim_state
        self.W1z = np.random.randn(layers, 1) 
        self.W1v = np.random.randn(layers, 1)
        self.b1z = np.zeros(layers)
        self.b1v = np.zeros(layers)
        self.WNz = np.abs(np.random.randn(1, layers))
        self.WNv = np.random.randn(1, layers)
        self.bNz = np.abs(np.random.randn(1))
        self.bNv = np.random.randn(1)
        self.epsilon = 0.1
        self.sub = 4


    def read_data(self,path):
        raw = np.loadtxt(path, delimiter=",", skiprows=0)
        self.M          = int(raw[:, 0].max()) + 1
        self.N          = raw.shape[0] // self.M
        self.time_grid  = raw[:self.N, 1]                        

        self.X = np.zeros((self.N,2,self.M))                        # (N,2,M)
        self.f2 = np.zeros((self.N,1,self.M))                       # (N,1,M)
        self.A = np.zeros((self.N,1,self.M))                        # (N,1,M)
        self.S = np.zeros((self.N,1,self.M))                        # (N,1,M)
        self.X_sub = np.zeros((self.N,self.sub,2,self.M))           # (N,sub=4,2,M)
        self.Z = np.zeros((self.N,1,self.M))                        # (N,1,M)
        self.Zq = np.zeros((self.N,1,self.M))                       # (N,1,M)
        self.M_hist = np.zeros((self.N,1,self.M))                   # (N,1,M)
        self.Mq = np.zeros((self.N,1,self.M))                       # (N,1,M)
        self.Mqq = np.zeros((self.N,1,self.M))                      # (N,1,M)
        self.Vq = np.zeros((self.N,1,self.M))                       # (N,1,M)
        self.Vqq = np.zeros((self.N,1,self.M))                      # (N,1,M)
        self.alphaZ = np.zeros((self.N,self.h,self.M))              # (N,h,M)
        self.betaZ = np.zeros((self.N,self.h,self.M))               # (N,h,M)
        self.alphaV = np.zeros((self.N,self.h,self.M))              # (N,h,M)
        self.betaV = np.zeros((self.N,self.h,self.M))               # (N,h,M)

        # Spalten: traj_id, t, q, qdot, u, qddot
        self.Q_start        = np.zeros(self.M)                      # (M,)
        self.Qdot_start     = np.zeros(self.M)                      # (M,)
        self.Q_data         = np.zeros((self.N,self.M))             # (N,M)
        self.Qdot_data      = np.zeros((self.N,self.M))             # (N,M)
        self.U              = raw[:, 4].reshape(self.M, self.N).T

        # Startwerte Speichern
        for k in range(self.M):
            self.Q_start[k]     = raw[k*self.N,2]
            self.Qdot_start[k]  = raw[k*self.N,3]
            for i in range(self.N):
                self.Q_data[i,k]        = raw[k*self.N+i,2]
                self.Qdot_data[i,k]     = raw[k*self.N+i,3]

        # vstack -> [ [Werte Q] ; [Werte Qdot] ]
        self.X[0] = np.vstack([self.Q_start, self.Qdot_start])      # (2,M)

    def forward_pass(self):
        for i in range(self.N):
            q    = self.X[i, 0]
            qdot = self.X[i, 1]

            (M, 
            zq, 
            A, 
            S, 
            Mq, 
            Mqq, 
            Vq, 
            Vqq, 
            alpha_z, 
            beta_z, 
            alpha_v, 
            beta_v) = self.lagrangian_term(q)

            self.M_hist[i] = M
            self.A[i]      = A
            self.S[i]      = S
            self.Mq[i]     = Mq
            self.Mqq[i]    = Mqq
            self.Zq[i]     = zq
            self.Vq[i]     = Vq
            self.Vqq[i]    = Vqq
            self.alphaZ[i] = alpha_z
            self.betaZ[i]  = beta_z
            self.alphaV[i] = alpha_v
            self.betaV[i]  = beta_v
            self.f2[i]     = (1 / M) * (-0.5 * Mq * qdot**2 - Vq + self.U[i])
            
            # Integration der ODE
            if i < self.N-1:
                dt = self.time_grid[i+1] - self.time_grid[i]
                step_size  = dt / self.sub
                q       = self.X[i, 0].copy()
                qdot    = self.X[i, 1].copy()
                u_i     = self.U[i]
                for k in range(self.sub):
                    q, qdot = self.RK4_step(q, qdot, u_i, step_size)
                    self.X_sub[i, k, 0] = q
                    self.X_sub[i, k, 1] = qdot
                self.X[i+1, 0] = q
                self.X[i+1, 1] = qdot

    #  Berechnung von q'' basierend auf q und q' im skalaren Fall:
    # q'' = 1/H * (L_q - C*q' + u) mit H=M, C=M'*q' L_q = 1/2 M' * (q')^2 - V'
    # => q'' = 1/M (-1/2 M' * (q')^2 - V' +u)
    def forward(self,q,qdot,u=0):
        # Berechnung von z im skalaren Fall: z = WN * log(1+exp(W1 q + b1)) + bN
        z = self.WNz@(np.log(1+np.exp(self.W1z[:,0]*q+self.b1z)))+self.bNz

        # Berechnung von A im skalren Fall: A= log(1+exp(z))
        A = np.log(1+np.exp(z))

        # Berechnung von M im skalaren Fall: M = A^2 + epsilon 
        M = A**2 + self.epsilon

        # Berechnung von M' = 2 A(q) s(q) * z'(q)
        S = 1/(1+np.exp(-z))
        alpha_z = np.exp(self.W1z[:,0]*q+self.b1z)/(1+np.exp(self.W1z[:,0]*q+self.b1z))
        Zq = self.WNz@(alpha_z*self.W1z[:,0])
        Mq = 2*A*S*Zq

        # Berechnung V' = W_N (alpha_V * W_1)
        alpha_v = np.exp(self.W1v[:,0]*q+self.b1v)/(1+np.exp(self.W1v[:,0]*q+self.b1v))
        Vq = self.WNv @ (alpha_v * self.W1v[:,0])

        return M[0],Mq[0], Vq[0]

    # Hier eine Funktion die immer eine volle Berechnung macht.
    # Hier später full als Option einfügen 
    def lagrangian_term(self,q):
        z_pre = self.W1z @ q[None, :] + self.b1z[:, None]       # (h, M)
        v_pre = self.W1v @ q[None, :] + self.b1v[:, None]       # (h, M)

        # softplus: Z=log(1+e^z_pre)
        sigma_z = np.log(1 + np.exp(z_pre))
        sigma_v = np.log(1 + np.exp(v_pre))
        alpha_z = np.exp(z_pre) / (1 + np.exp(z_pre))
        beta_z  = alpha_z * (1 - alpha_z)
        alpha_v = np.exp(v_pre) / (1 + np.exp(v_pre))
        beta_v  = alpha_v * (1 - alpha_v)

        Z   = self.WNz @ sigma_z + self.bNz[:, None]            # (1, M)
        A   = np.log(1+np.exp(Z))                               # (1, M)
        S   = 1/(1+np.exp(-Z))                                  #
        M   = A*A + self.epsilon                                # (1, M)
        zq  = self.WNz @ (alpha_z * self.W1z)                   # (1, M)
        zqq = self.WNz @ (beta_z * self.W1z**2)
        Mq  = 2*A*S*zq
        Mqq = 2*(S**2 * zq**2 + A*S*(1-S)* zq**2 + A*S*zqq)     # (1, M)
        Vq  = self.WNv @ (alpha_v * self.W1v)
        Vqq = self.WNv @ (beta_v * self.W1v**2)

        # M[0] -> Rückgabe als Vektor (M,)
        return (
            M[0], 
            zq[0], 
            A[0], 
            S[0], 
            Mq[0], 
            Mqq[0], 
            Vq[0], 
            Vqq[0], 
            alpha_z, 
            beta_z, 
            alpha_v, 
            beta_v) # Rückgabe 12 Werte

    # f2 = 1/M * (-1/2 M' * (q')^2 - V'(q) + u)
    def f(self, q, qdot, u):
        M, _ ,_ , _ , Mq ,_, Vq, _ , _ , _ , _ , _ = self.lagrangian_term(q)
        qddot = (-1/2 * Mq * (qdot**2) - Vq + u) / M
        return qdot, qddot

    # Runge-Kutta Ordnung 4:
    # k1, k1' = f(q,q')
    # k2, k2' = f(q + h/2 k1, q' + h/2 k1')
    # k3, k3' = f(q + h/2 k2, q' + h/2 k2')
    # k4, k4' = f(q + h k3, q' + h k3')
    # q_new = q + h/6 (k1 + 2k2 + 2k3 + k4)
    # q'_new = q' + h/6 (k1' + 2k2' + 2k3' + k4')
    
    def RK4_step(self, q, qdot, u, step_size):
        k1q, k1qdot = self.f(q, qdot, u)
        k2q, k2qdot = self.f(q + 0.5*step_size*k1q, qdot + 0.5*step_size*k1qdot, u)
        k3q, k3qdot = self.f(q + 0.5*step_size*k2q, qdot + 0.5*step_size*k2qdot, u)
        k4q, k4qdot = self.f(q + step_size*k3q, qdot + step_size*k3qdot, u)
        q_new = q + step_size/6 * (k1q + 2*k2q + 2*k3q + k4q)
        qdot_new = qdot + step_size/6 * (k1qdot + 2*k2qdot + 2*k3qdot + k4qdot)
        return q_new, qdot_new

    def compute_f_x_single(self, i, p1, p2):
        # Auswertung von f_x * P an beliebigen Werten von P wegen integration
        q               = self.X[i,0]
        qdot            = self.X[i,1]       
        M, _ , _ , _ , Mq, Mqq, _, Vqq, _, _, _, _  = self.lagrangian_term(q)

        # Erste Komponente von (f_x)*P:
        # p_2 *  (-1)* ((V''+ 1/2 M'' q'^2)/M + f_2 M'/M)
        fxP_1 = p2* (
            -(Vqq + 1/2 * Mqq * qdot**2)/M - self.f2[i][0]*(Mq/M)
        )
        # Zweite Komponente von (f_x)*P:
        # p_1 - (M'/M)q' p_2
        fxP_2 = p1 - p2*(Mq/M)*qdot 
        return fxP_1, fxP_2

    def backward_pass(self):
        self.P = np.zeros((self.N,2,self.M))
        for i in range(self.N-1,0,-1):
            dt  = self.time_grid[i] - self.time_grid[i-1]
            p1 = self.P[i, 0].copy()
            p2 = self.P[i, 1].copy()
            res_q       = 2 * (self.X[i, 0] - self.Q_data[i])   # Loss-Gradient
            res_qdot    = np.zeros_like(res_q)

            def rhs(p1, p2):
                fxP_1, fxP_2 = self.compute_f_x_single(i, p1, p2)
                return -fxP_1 - res_q, -fxP_2 - res_qdot

            k1q, k1qdot = rhs(p1,             p2)
            k2q, k2qdot = rhs(p1 - 0.5*dt*k1q, p2 - 0.5*dt*k1qdot)
            k3q, k3qdot = rhs(p1 - 0.5*dt*k2q, p2 - 0.5*dt*k2qdot)
            k4q, k4qdot = rhs(p1 - dt*k3q,     p2 - dt*k3qdot)
            self.P[i-1, 0] = p1 - dt/6 * (k1q + 2*k2q + 2*k3q + k4q)
            self.P[i-1, 1] = p2 - dt/6 * (k1qdot + 2*k2qdot + 2*k3qdot + k4qdot)


    def compute_gradient(self):
        # Broadcast der Gewichte:
        WNz = (self.WNz.T) @ np.ones((1,self.M))                        # (h,M)
        WNv = (self.WNv.T) @ np.ones((1,self.M))                        # (h,M)
        W1z = self.W1z @ np.ones((1,self.M))                            # (h,M)
        W1v = self.W1v @ np.ones((1,self.M))                            # (h,M)
        b1z = self.b1z[:, None] @ np.ones((1,self.M))
        b1v = self.b1v[:, None] @ np.ones((1,self.M))
        bNz = self.bNz[0] * np.ones((1,self.M))

        # Anlegen der Arrays für den Gradienten:
        grad_W1z = np.zeros(self.h)                                     # (h,)
        grad_b1z = np.zeros(self.h)                                     # (h,)
        grad_WNz = np.zeros(self.h)                                     # (h,)
        grad_W1v = np.zeros(self.h)                                     # (h,)
        grad_b1v = np.zeros(self.h)                                     # (h,)
        grad_WNv = np.zeros(self.h)                                     # (h,)
        grad_bNz = np.zeros(1)                                          # (1,)

        dt = self.time_grid[1] - self.time_grid[0]
        for i in range(self.N):
            q       = np.ones((self.h,1)) @ self.X[i,0][None,:]         # (h,M)
            qdot    = np.ones((self.h,1)) @ self.X[i,1][None,:]         # (h,1)*(1,M)=(h,M)
            P2      = self.P[i,1]

            # Block mit den ABleitungen dz/dtheta:
            #----------------------------------------------------------------------
            # dz/dW1z = (W^N)^T * alpha_z * q
            dzdW1   = (WNz * self.alphaZ[i]) * q
            # dz/db1 = (W^N)^T * alpha_z 
            dzdb1   = WNz * self.alphaZ[i]
            z_pre = W1z * q + b1z                                       # (h, M)
            dzdWN   = np.log(1+ np.exp(z_pre))
            dzdbN   = np.ones((1,self.M))

            dzqdW1  = WNz * (self.betaZ[i] * (W1z * q) + self.alphaZ[i])
            dzqdb1  = WNz * (self.betaZ[i] * W1z)
            dzqdWN  = self.alphaZ[i]*W1z
            dzqdbN  = np.zeros((1,self.M))

            dVqdW1  = WNv * (self.betaV[i]*W1v*q + self.alphaV[i])
            dVqdWN  = self.alphaV[i]*W1v
            dVqdb1  = WNv * (self.betaV[i] * W1v)

            M = self.M_hist[i]
            Mq = self.Mq[i]
            Vq = self.Vq[i]
            u = self.U[i]
            A = self.A[i]
            S = self.S[i]
            Zq = self.Zq[i]

            # dM/dtheta = 2A(q) s(q) dz/dtheta:
            #---------------------------------------------------------------
            dMdW1 = 2*A*S*dzdW1
            dMdb1 = 2*A*S*dzdb1
            dMdWN = 2*A*S*dzdWN
            dMdbN = 2*A*S*dzdbN

            # dM'/dtheta = 2(s(q)^2 + A(q)s(q)(1-s(q)) dz/dtheta * z' + A(q)s(q) dz'/dtheta)
            #---------------------------------------------------------------
            dMqdW1 = 2*((S**2 + A*S*(1-S))*dzdW1*Zq + A*S*dzqdW1)
            dMqdb1 = 2*((S**2 + A*S*(1-S))*dzdb1*Zq + A*S*dzqdb1)
            dMqdWN = 2*((S**2 + A*S*(1-S))*dzdWN*Zq + A*S*dzqdWN)
            dMqdbN = 2*((S**2 + A*S*(1-S))*dzdbN*Zq + A*S*dzqdbN)

            # Block (-1/M dV'/dtheta_V)*p_2
            grad_W1v += dt * ((-1/M)*dVqdW1) @ P2 
            grad_b1v += dt * ((-1/M)*dVqdb1) @ P2 
            grad_WNv += dt * ((-1/M)*dVqdWN) @ P2 

            # [-1/M^2 dM/dtheta * (-1/2 Mq X_2^2 - Vq + U) + 1/M (-1/2 X_2^2 dM'/dtheta)]*p_2
            grad_W1z += dt * (
                -1/M**2 * dMdW1 * (
                    -1/2 *Mq *qdot**2 - Vq + u
                    ) + 1/M*(
                        -1/2*qdot**2 * dMqdW1
                    )
                ) @ P2

            grad_b1z += dt * (
                -1/M**2 * dMdb1 * (
                    -1/2 *Mq *qdot**2 - Vq + u
                    ) + 1/M*(
                        -1/2*qdot**2 * dMqdb1
                    )
                ) @ P2

            grad_WNz += dt * (
                -1/M**2 * dMdWN * (
                    -1/2 *Mq *qdot**2 - Vq + u
                    ) + 1/M*(
                        -1/2*qdot**2 * dMqdWN
                    )
                ) @ P2

            # Wichtig qdot ist (h,M) aber wir brauchen hier (1,M)
            grad_bNz += dt * (
                -1/M**2 * dMdbN * (
                    -1/2 *Mq * self.X[i, 1][None, :]**2 - Vq + u
                    ) + 1/M*(
                        -1/2* self.X[i, 1][None, :]**2 * dMqdbN
                    )
                ) @ P2

        return (grad_W1v/(self.N*self.M), 
                grad_b1v/(self.N*self.M), 
                grad_WNv/(self.N*self.M), 
                grad_W1z/(self.N*self.M), 
                grad_b1z/(self.N*self.M),
                grad_WNz/(self.N*self.M),
                grad_bNz/(self.N*self.M)
        )
    
    def train_step(self,learning_rate= 0.01):
        self.forward_pass()
        self.backward_pass()
        (   grad_W1v, 
            grad_b1v,
            grad_WNv,
            grad_W1z, 
            grad_b1z,
            grad_WNz,
            grad_bNz) = self.compute_gradient()
    
        self.W1z -= learning_rate * grad_W1z[:,None]        # (h,)
        self.b1z -= learning_rate * grad_b1z                # (h,)
        self.WNz -= learning_rate * grad_WNz                # (1,h)
        self.bNz -= learning_rate * grad_bNz                # (1,)

        self.W1v -= learning_rate * grad_W1v[:,None]
        self.b1v -= learning_rate * grad_b1v
        self.WNv -= learning_rate * grad_WNv

    def loss(self):
        # sum_i (q-q_data)^2 
        error = self.X[:,0,:] - self.Q_data                 # (N,M)
        #dt = self.time_grid[1] - self.time_grid[0]
        return float(np.mean((error**2)))

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