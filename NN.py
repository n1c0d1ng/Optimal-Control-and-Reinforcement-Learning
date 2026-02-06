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

    
    # Wichtig flatten(): Geht Zeilenweise vor: 
    # [[1,2],[3,4]] -> [1,2,3,4]
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

# z_1 = W_1 * [t \\ x] + b_1 
# a_1 = tanh(z_1)
# z_2 = W_2 * a_1 + b_2
# a_2 = tanh(z_2)
# z_N = W_N a_2 + b_N
class Neural_Network_Two_Layer:
    def __init__(self, n1=3 , n2 =3):
        self.n1 = n1
        self.n2 = n2
    
    def ForwardPass(self, t, X):
        return 0
    
    def Backpropagation(self, t, X):
        return 0