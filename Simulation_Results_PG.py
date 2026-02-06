# Ziel dieser Datei:
# 1. Berechnung der optimalen Lösung mittels Adjoint Gradient Descent
# 2. Berechnung der Policy mittels Policy Gradient mit einem Neural Network
# 3. Simulation der gelernten Policy und Speichern in CSV-Dateien

import numpy as np
import matplotlib.pyplot as plt
from ControlProblem import OCP
from NN import Neural_Network_One_Layer
from Policy_Gradient_NN import Train_Policy_Gradient

show_plots = False

Environment = OCP(N=100)

# Hier werden die numerischen Lösungen mittels Adjoint Gradient Descent berechnet:
Environment.gradient_descent(max_iter=1000, tol=1e-6)
Policy = Neural_Network_One_Layer(n=5, input_dim=2, output_dim=1)

for i in range(2):
    Parameters = Train_Policy_Gradient(Environment, num_episodes=50000)
    Policy.set_parameters(Parameters)

    # Simulation der gelernten Policy
    t_test, x_test = 0.0, 0.0
    trajectory_learned = np.zeros(Environment.N + 1)
    trajectory_learned[0] = x_test
    u_learned = np.zeros(Environment.N)

    # Simulation der Ergebnisse basierend auf der gelernten Policy
    for step in range(Environment.N):
        # Kein Rauschen - deterministisch! mittels Euler-Diskretisierung
        u_optimal = Policy.ForwardPass(Environment.t[step], trajectory_learned[step])[0]
        x_test = Environment.get_next_state(x_test, u_optimal)
        trajectory_learned[step+1] = x_test
        u_learned[step] = u_optimal

    # Ergebnisse der Simulation festhalten: 
    f = open('Simulation_Control.csv', 'a')
    np.savetxt(f, u_learned.reshape(1, Environment.N), delimiter=',')
    f.close()

    f = open('Simulation_Trajectory.csv', 'a')
    np.savetxt(f, trajectory_learned.reshape(1, Environment.N + 1), delimiter=',')
    f.close()

if show_plots == True:
    # Plot: Gelernte vs. Optimale Lösung
    plt.figure(figsize=(10, 6))

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