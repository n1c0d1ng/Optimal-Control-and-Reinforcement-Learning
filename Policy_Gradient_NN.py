from ControlProblem import OCP
from NN import Neural_Network_One_Layer
import numpy as np

def Train_Policy_Gradient(Environment, num_episodes=50000, learning_rate=0.0001, exploration=1.0):
    #Environment = OCP(N=100)

    # Anlegen Neural Network 
    Policy = Neural_Network_One_Layer(n=5, input_dim=2, output_dim=1)
    sigma = exploration  # Standardabweichung für die Rauschkomponente
    alpha = learning_rate 

    for episode in range(num_episodes):
        x, t = 0.0, 0.0
        trajectory = np.zeros(Environment.N + 1)
        time_grid = np.zeros(Environment.N + 1)
        trajectory[0] = x
        time_grid[0] = t
        rewards = np.zeros(Environment.N + 1)
        actions = np.zeros(Environment.N)

        # Interaktion mit der Environment
        for step in range(Environment.N):
            # Policy aus NN
            mu = Policy.ForwardPass(time_grid[step], trajectory[step])[0]
            
            # u sim pi(u|t,x) = N(mu, sigma^2)
            actions[step] = np.clip(np.random.normal(mu, sigma), -10, 10)
            rewards[step] = Environment.direct_reward(x, actions[step])  
            x_next = Environment.get_next_state(x, actions[step])
            t_next = t + Environment.h
            t, x = t_next, x_next
            trajectory[step+1] = x
            time_grid[step+1] = t

        R = np.zeros(Environment.N + 1)
        rewards[Environment.N] = Environment.terminal_reward(trajectory[Environment.N])
        Sum = 0.0
        for i in range(Environment.N, -1, -1):  # Rückwärts durchlaufen
            Sum += rewards[i]
            R[i] = Sum
        
        grad = np.zeros(Policy.parameter_dim)
        
        # J_theta approx sum_{i=1}^{N} R(t_i)*log pi_theta(u(t_i)|t_i,X_i)] 
        for k in range(Environment.N):
            # pi = 1/(sqrt(2pi)*sigma)* exp[-(u - mu)^2/(2*sigma^2)] 
            # -> log pi = -0.5*log(2pi*sigma^2) - (u - mu_theta)^2/(2*sigma^2)
            # grad_theta log(pi) = (u - mu) / sigma^2 * (dmu/dtheta)
            # dmu/dtheta = Backpropagation des NN
            mu = Policy.ForwardPass(time_grid[k], trajectory[k])[0]
            grad_logp = (actions[k] - mu) / sigma**2 * Policy.Backpropagation(time_grid[k], trajectory[k])
            grad += R[k] * grad_logp

        #if np.linalg.norm(grad) > 1:
        #    grad = grad / np.linalg.norm(grad)  # Gradienten-Norm begrenzen

        sigma = max(0.1, sigma * 0.99998)  # Exploration reduzieren
        Policy.set_parameters(Policy.get_parameters() + alpha * grad)
        Policy.set_parameters(np.clip(Policy.get_parameters(), -100, 100))
    return Policy.get_parameters()