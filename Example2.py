# Implementierung von
# min int_0^1 u^2 dt + (x(1)-1)^2  mit x'=u x(0) = 0
#--------------------------------------------------------------------------------------------------
import ControlProblem as OCP
import matplotlib.pyplot as plt
import numpy as np

problem = OCP.SimpleControlProblem(N=100)
#problem.gradient_descent(max_iter=1000, tol=1e-6)

# Policy Gradient
theta = np.array([0.0])  # Initiale Parameter [theta_0]
sigma = 1.0  # Standardabweichung für die Rauschkomponente
alpha = 0.0001  # Lernrate

num_episodes = 50000
for episode in range(num_episodes):
    x, t = 0.0, 0.0
    trajectory = []
    rewards = []
    actions = []

    for step in range(100):
        # Policy aus dem theoretischen Setting: u = A
        mu = theta[0]
        
        # u sim pi(u|t,x) = N(mu, sigma^2)
        u = np.clip(np.random.normal(mu, sigma), -10, 10)
        r = -(u**2)*(1/100)  # Sofortige Belohnung

        trajectory.append((t, x))
        rewards.append(r)
        actions.append(u)

        # Das kennt der Algorithmus nicht
        x_next = x + (1/100)*(u)
        t_next = t + (1/100)
        t, x = t_next, x_next

    # Gradientenschritt basierend auf dieser einen trajektorie
    rewards.append(-(x-1)**2)  # Endterm als Reward (negativ, da Minimierung)
    trajectory.append((t, x))
    #actions.append(u)

    R = np.zeros(100)

    for i in range(100):
        R[i] = sum(rewards[i:])  # Gesamtrückgabe ab Zeit i
        
    grad = np.zeros(1)
    for k in range(100):
        # mu = theta_0
        mu = theta[0]

        # pi = 1/(sqrt(2pi)*sigma)* exp[-(u - mu)^2/(2*sigma^2)] 
        # -> log pi = -0.5*log(2pi*sigma^2) - (u - mu_theta)^2/(2*sigma^2)
        # grad_theta log(pi) = (u - mu) / sigma^2 * (dmu/dtheta)
        # dmu/dtheta = [1]
        grad_logp = (actions[k] - mu) / sigma**2 * np.array([1.0])
        grad += R[k] * grad_logp
    sigma = max(0.2, sigma * 0.9998)  # Exploration reduzieren
        
    theta += alpha * grad
    theta = np.clip(theta, -100, 100)

# Ausgabe der gelernten Parameter
# Teste die gelernte Policy OHNE Exploration
x_test, t_test = 0.0, 0.0
trajectory_learned = []
u_learned = [] 


for step in range(100):
    u_optimal = theta[0]  # Kein Rauschen - deterministisch!
    x_test = x_test + (1/100) * u_optimal
    t_test = t_test + (1/100)
    
    trajectory_learned.append((t_test, x_test))
    u_learned.append(u_optimal)


results = np.array(trajectory_learned)
control = np.array(u_learned)
plt.plot(results[:, 0], results[:, 1], label="Trajektorie")
plt.plot(results[:, 0], control, label="u numerisch")


#plt.plot(problem.t, problem.x, label="x numerisch")
plt.plot(problem.t, problem.x_analytic(), label="x analytisch")
#plt.plot(problem.t, problem.u, label="u numerisch")
plt.plot(problem.t, problem.u_analytic(), label="u analytisch")
plt.legend()
plt.show()
