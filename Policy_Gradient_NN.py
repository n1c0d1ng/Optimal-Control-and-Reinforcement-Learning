from ControlProblem import OCP
from NN import Neural_Network_One_Layer
import numpy as np

# Simulation einer Episode basierend auf der aktuellen Policy
def simulate_epoch(
        Environment, 
        Policy, 
        initial_state = 0.0,
        sigma = 0.0, 
        deterministic=False):
    N = Environment.N
    trajectory = np.zeros(N + 1)
    time_grid  = np.zeros(N + 1)
    actions    = np.zeros(N)
    rewards    = np.zeros(N + 1)

    # Für das allgemeine Problem: mit x0  beliebig
    trajectory[0] = initial_state
    time_grid[0]  = 0.0

    for step in range(N):
        mu = Policy.ForwardPass(np.array([time_grid[step], trajectory[step]]))[0]
        if deterministic:
            u = mu
        else:
            u = np.clip(np.random.normal(mu, sigma), -10, 10)
        actions[step] = u
        rewards[step] = Environment.direct_reward(trajectory[step], u)
        trajectory[step + 1] = Environment.get_next_state(trajectory[step], u)
        time_grid[step + 1] = time_grid[step] + Environment.h

    rewards[N] = Environment.terminal_reward(trajectory[N])
    return trajectory, time_grid, actions, rewards

# Ausgabe des Zielfunktionals basierend auf den aktuellen Rewards
def episode_cost(rewards):
    return -float(np.sum(rewards))


def evaluate_policy(Environment, Policy , initial_state = 0.0):
    (traj,
    _, 
    actions, 
    rewards) = simulate_epoch(Environment, Policy, initial_state ,sigma=0.0, deterministic=True)
    return traj, actions, episode_cost(rewards)

# Update der Policy Parameter
def train_step(Environment, Policy, sigma, alpha, initial_state =0.0, clip=100.0):
    N = Environment.N

    # Simulation einer Trajektorie
    #--------------------------------------------------
    (trajectory, 
    time_grid,
    actions, 
    rewards) = simulate_epoch(Environment, Policy, initial_state, sigma)
    R = np.zeros(N + 1)
    running = 0.0

    # Rewards ab Zeitpunkt i bis zum Ende
    for i in range(N, -1, -1):
        running += rewards[i]
        R[i] = running

    # Gradientenschaetzung
    grad = np.zeros(Policy.parameter_dim)
    for k in range(N):
        # pi = 1/(sqrt(2pi)*sigma)* exp[-(u - mu)^2/(2*sigma^2)] 
        # -> log pi = -0.5*log(2pi*sigma^2) - (u - mu_theta)^2/(2*sigma^2)
        # grad_theta log(pi) = (u - mu) / sigma^2 * (dmu/dtheta)
        # dmu/dtheta = Backpropagation des NN
        mu = Policy.ForwardPass(np.array([time_grid[k], trajectory[k]]))[0]
        dmu_dtheta = Policy.Backpropagation(np.array([time_grid[k], trajectory[k]]))
        grad_logp = (actions[k] - mu) / sigma**2 * dmu_dtheta
        grad += R[k] * grad_logp

    # Parameterupdate
    new_params = np.clip(Policy.get_parameters() + alpha * grad, -clip, clip)
    Policy.set_parameters(new_params)
    return episode_cost(rewards)


def train_step_pathwise(
        Environment, 
        Policy, 
        sigma, 
        alpha, 
        initial_state = 0.0 , 
        clip=100.0):
    
    N = Environment.N
    h = Environment.h

    # Forward Pass:
    #----------------------------------------------------------------------------
    trajectory = np.zeros(N + 1)
    time_grid  = np.zeros(N + 1)
    actions    = np.zeros(N)
    mu_x_hist  = np.zeros(N)        # dmu/dx an (t_k, x_k)
    rewards    = np.zeros(N + 1)

    trajectory[0] = initial_state
    time_grid[0]  = 0.0

    for k in range(N):
        # mu(ti,xi) = NN
        mu  = Policy.ForwardPass(np.array([time_grid[k], trajectory[k]]))[0]

        # u -> N(mu,sigma^2)
        u   = np.clip(np.random.normal(mu, sigma), -10, 10)
        actions[k] = u

        # Sensitivitaet dmu/dx (Komponente 1 des Gradienten nach dem Input)
        mu_x_hist[k] = Policy.sensitivities(np.array([time_grid[k], trajectory[k]]))[1]

        # Interaktion mit der Umgebung
        #----------------------------------------------------------------------
        rewards[k] = Environment.direct_reward(trajectory[k], u)
        trajectory[k+1] = Environment.get_next_state(trajectory[k], u)
        time_grid[k+1] = time_grid[k] + h

    rewards[N] = Environment.terminal_reward(trajectory[N])

    # Backward Pass:
    p = np.zeros(N + 1)
    p[N] = 2.0 * trajectory[N]      # dPhi/dx = 2x_N

    for k in range(N - 1, -1, -1):
        L_u  = 2.0 * actions[k]
        mu_x = mu_x_hist[k]
        # p' = -(L_x + L_u dmu/dx) - (f_x + f_u dmu/dx) * p
        # L_x + L_u*mu_x + (f_x + f_u*mu_x) * p_k  mit L_x=f_x=f_u=1
        p[k] = p[k + 1] + h * ((1.0 + L_u * mu_x) + (1.0 + mu_x) * p[k + 1])

    # Gradient:
    grad = np.zeros(Policy.parameter_dim)
    for k in range(N):
        # L(x,u) = x+u^2 -> L_u = 2u, L_x = 1
        L_u  = 2.0 * actions[k]
        # nabla J = E[int_0^T dmu/dtheta * (L_u + p*f_u) dt]
        # (L_u + p_k * f_u) * dmu/dtheta  mit f_u = 1
        grad += h * (L_u + p[k]) * Policy.Backpropagation(
            np.array([time_grid[k], trajectory[k]]))

    # Update Schritt
    #--------------------------------------------------------------------------
    new_params = np.clip(Policy.get_parameters() - alpha * grad, -clip, clip)
    Policy.set_parameters(new_params)

    return episode_cost(rewards)   