# Implementierung von
# min int_0^1 x+u^2 dt + x(1)^2  mit x'=x+u+1 x(0) = 0
#--------------------------------------------------------------------------------------------------
from ControlProblem import OCP
import matplotlib.pyplot as plt
import numpy as np

Environment = OCP(N=100)

# Hier werden die numerischen Lösungen mittels Adjoint Gradient Descent berechnet:
Environment.gradient_descent(max_iter=1000, tol=1e-6)

# Policy Gradient
theta = np.array([-1.0, 1.5, 1.0])  # Parameter [A, B, C]
sigma = 1  # Standardabweichung für die Rauschkomponente
alpha = 0.0001  # Lernrate

num_episodes = 100000
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
        # Policy aus dem theoretischen Setting: u = A * exp(-B*t) + C
        mu = theta[0]* np.exp(-theta[1]*time_grid[step]) + theta[2]
        
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
        
    grad = np.zeros(3)
    for k in range(Environment.N):
        # mu = theta_0 * exp(- theta_1 *t) + theta_2
        mu = theta[0]* np.exp(-theta[1]*time_grid[k]) + theta[2]

        # pi = 1/(sqrt(2pi)*sigma)* exp[-(u - mu)^2/(2*sigma^2)] 
        # -> log pi = -0.5*log(2pi*sigma^2) - (u - mu_theta)^2/(2*sigma^2)
        # grad_theta log(pi) = (u - mu) / sigma^2 * (dmu/dtheta)
        # dmu/dtheta = [exp(- theta_1 *t) , -theta_0*t*exp(- theta_1 *t), 1]
        grad_logp = (actions[k] - mu) / sigma**2 * \
              np.array(
                    [
                        np.exp(-theta[1]*time_grid[k]),
                        -theta[0]*time_grid[k]*np.exp(-theta[1]*time_grid[k]),
                        1
                    ]
                )
        grad += R[k] * grad_logp
    sigma = max(0.1, sigma * 0.99998)  # Exploration reduzieren
    theta += alpha * grad
    theta = np.clip(theta, -100, 100)

# Simulation der gelernten Policy
print(f"Gelearnte Parameter theta[0] = {theta[0]}, theta[1] = {theta[1]}, theta[2] = {theta[2]}")
print(f"Optimale Parameter theta[0] = {-(3*np.exp(2) - 2*np.exp(1)) / (np.exp(2) + 1)}, theta[1] = 1, theta[2] = 0.5")
t_test, x_test = 0.0, 0.0
trajectory_learned = np.zeros(Environment.N + 1)
trajectory_learned[0] = x_test
u_learned = np.zeros(Environment.N)

# Simulation der Ergebnisse basierend auf der gelernten Policy
for step in range(Environment.N):
    # Kein Rauschen - deterministisch! mittels Euler-Diskretisierung
    u_optimal = theta[0]* np.exp(-theta[1]*Environment.t[step]) + theta[2]  
    x_test = Environment.get_next_state(x_test, u_optimal)
    trajectory_learned[step+1] = x_test
    u_learned[step] = u_optimal


results = np.array(trajectory_learned)
control = np.array(u_learned)

# Plot: Gelernte vs. Optimale Lösung
plt.figure(figsize=(12, 8))

# Subplot 1: Zustand
plt.subplot(2, 1, 1)
plt.plot(Environment.t, results, 'b-', label="State (PG)", linewidth=2)
plt.plot(Environment.t, Environment.x, 'r--', label="State (Optimal)", linewidth=2)
plt.plot(Environment.t, Environment.x_analytic(), 'g:', label="State (Analytic)", linewidth=2)
plt.xlabel('Time t')
plt.ylabel('State x(t)')
plt.title('Policy Gradient vs. Optimal Control - State')
plt.legend()
plt.grid(True)

# Subplot 2: Steuerung
plt.subplot(2, 1, 2)
# Achtung: control hat Länge N, Environment.t[:-1] hat auch Länge N
plt.plot(Environment.t[:-1], control, 'b-', label="Policy (PG)", linewidth=2)  
plt.plot(Environment.t, Environment.u, 'r--', label="Control (Optimal)", linewidth=2)
plt.plot(Environment.t, Environment.u_analytic(), 'g:', label="Control (Analytic)", linewidth=2)
plt.xlabel('Time t')
plt.ylabel('Control u(t)')
plt.title('Policy Gradient vs. Optimal Control - Control')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()