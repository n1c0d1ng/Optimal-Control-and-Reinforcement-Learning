# Implementierung von
# min int_0^1 x+u^2 dt x'=x+u+1 x(0) = 0
#--------------------------------------------------------------------------------------------------
import ControlProblem as OCP
import matplotlib.pyplot as plt
import numpy as np

problem = OCP.ControlProblem(N=100)
problem.gradient_descent(max_iter=1000, tol=1e-6)

# Policy Gradient
theta = np.array([0.0, 0.0, 0.0])  # Initiale Parameter [theta_0, theta_1]
sigma = 1  # Standardabweichung für die Rauschkomponente
alpha = 0.0001  # Lernrate

num_episodes = 100000000
for episode in range(num_episodes):
    x, t = 0.0, 0.0
    trajectory = []
    rewards = []
    actions = []
    trajectory.append((t, x))
    actions.append(0)

    for step in range(100):
        mu = theta[0] + theta[1]*t + theta[2]*x
        # u sim pi(u|t,x) = N(mu, sigma^2)
        u = np.clip(np.random.normal(mu, sigma), -10, 10)
        x_next = x + (1/100)*(x + u + 1)
        t_next = t + (1/100)
        r = -(x + u**2)*(1/100)  # Sofortige Belohnung

        trajectory.append((t, x))
        rewards.append(r)
        actions.append(u)
        t, x = t_next, x_next

    # Gradientenschritt basierend auf dieser einen trajektorie
    rewards.append(-x**2)  # Endterm als Reward (negativ, da Minimierung)
    R = np.zeros(100)

    for i in range(100):
        R[i] = sum(rewards[i:])  # Gesamtrückgabe ab Zeit 
        
    grad = np.zeros(3)
    for k in range(100):
        # mu = theta_0 + theta_1*t_k + theta_2*x_k
        mu = theta[0] + theta[1]*trajectory[k][0] + theta[2]*trajectory[k][1]

        # pi = N(mu, sigma^2) -> log pi = -0.5*log(2pi*sigma^2) - (u - mu)^2/(2*sigma^2)
        # grad log(pi) = (u - mu) / sigma^2 * [1, t_k, x_k]
        # da mu_theta = [1, t_k, x_k]
        grad_logp = (actions[k] - mu) / sigma**2 * np.array([1, trajectory[k][0], trajectory[k][1]])
        grad += R[k] * grad_logp
    sigma = max(0.2, sigma * 0.998)  # Exploration reduzieren
    if np.linalg.norm(grad) > 1:
        grad = grad/np.linalg.norm(grad)  # Normalisieren
        
    theta += alpha * grad
    theta = np.clip(theta, -100, 100)

results = np.array(trajectory)
control = np.array(actions)
plt.plot(results[:, 0], results[:, 1], label="Trajektorie")
plt.plot(results[:, 0], control, label="u numerisch")


#plt.plot(problem.t, problem.x, label="x numerisch")
plt.plot(problem.t, problem.x_analytic(), label="x analytisch")
#plt.plot(problem.t, problem.u, label="u numerisch")
plt.plot(problem.t, problem.u_analytic(), label="u analytisch")
plt.legend()
plt.show()

