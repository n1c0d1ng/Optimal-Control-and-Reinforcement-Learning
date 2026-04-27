# Actor-Critic Implementierung für SCOP:
# min_pi int_0^1 (X_t + u_t^2) dt + X_1^2

from NN import Neural_Network_One_Layer
import numpy as np


def Train_Actor_Critic(Environment, num_episodes=10000, learning_rate=0.0001, exploration=1.0):

    # Anlegen Neural Network 
    Policy = Neural_Network_One_Layer(n=5, input_dim=2, output_dim=1)
    Value_Function = Neural_Network_One_Layer(n=10,input_dim=2, output_dim= 1)
    sigma = exploration  
    alpha = learning_rate 
    beta = alpha*10

    for episode in range(num_episodes):
        x, t = 0.0, 0.0
        trajectory = np.zeros(Environment.N + 1)
        time_grid = np.zeros(Environment.N + 1)
        trajectory[0] = x
        time_grid[0] = t
        rewards = np.zeros(Environment.N + 1)
        actions = np.zeros(Environment.N)

        # Simulation einer Trajektorie
        for step in range(Environment.N):
            # Policy aus NN
            u = Policy.ForwardPass(np.array([time_grid[step], trajectory[step]]))[0]
            
            # u sim pi(u|t,x) = N(mu, sigma^2)
            actions[step] = np.clip(np.random.normal(u, sigma), -10, 10)
            rewards[step] = Environment.direct_reward(x, actions[step])  
            x_next = Environment.get_next_state(x, actions[step])
            t_next = t + Environment.h
            t, x = t_next, x_next
            trajectory[step+1] = x
            time_grid[step+1] = t

        # Abspeichern Rewards ab zeitpunkt t_i
        R = np.zeros(Environment.N + 1)
        rewards[Environment.N] = Environment.terminal_reward(trajectory[Environment.N])
        Sum = 0.0
        for i in range(Environment.N, -1, -1):  # Rückwärts durchlaufen
            Sum += rewards[i]
            R[i] = Sum
        
        grad_Actor = np.zeros(Policy.parameter_dim)
        grad_Critic = np.zeros(Value_Function.parameter_dim)
    

        for k in range(Environment.N):
            u = Policy.ForwardPass(np.array([time_grid[k], trajectory[k]]))[0]
            V_val = Value_Function.ForwardPass(np.array([time_grid[k], trajectory[k]]))[0]
            
            grad_logp = (actions[k] - u) / sigma**2 * Policy.Backpropagation(np.array([time_grid[k], trajectory[k]]))
            grad_Actor += (R[k] - V_val) * grad_logp

        for step in range(Environment.N +1):
            V_val = Value_Function.ForwardPass(np.array([time_grid[k], trajectory[k]]))[0]
            gradV = Value_Function.Backpropagation(np.array([time_grid[k], trajectory[k]]))
            grad_Critic += (V_val - R[k])*gradV


        sigma = max(0.1, sigma * 0.99998)  # Exploration reduzieren
        Policy.set_parameters(Policy.get_parameters() + alpha * grad_Actor)
        Value_Function.set_parameters(Value_Function.get_parameters() - beta*grad_Critic)
        Policy.set_parameters(np.clip(Policy.get_parameters(), -100, 100))

    return Policy.get_parameters()