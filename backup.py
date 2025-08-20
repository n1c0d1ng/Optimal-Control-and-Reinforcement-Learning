import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------------------
# 1. Parameterdefinition
# ------------------------------
params = {
    'm1': 1.0,       # Masse von Link 1 in kg
    'm2': 1.0,       # Masse von Link 2 in kg
    'l1': 1.0,       # Länge von Link 1 in m
    'l2': 1.0,       # Länge von Link 2 in m
    'g': 9.81        # Erdbeschleunigung in m/s^2
}

    # Farbdefinitionen
custom_background_dark = (53/255, 62/255, 69/255)       # Hintergrund der Graphik
custom_background_light = (1.0, 1.0, 1.0)                # Beschriftung & Koordinatensystem
custom_red = (255/255, 62/255, 123/255)                  # Link 2 & θ₂
custom_green = (60/255, 179/255, 113/255)                # Link 1 & θ₁

# ------------------------------
# 2. Konstante Drehmomentfunktion tau
# ------------------------------
def tau(N=1000, t=None):
    """
    Erzeugt einen Zeitvektor t (falls nicht vorhanden) und gibt 
    für jeden Zeitpunkt einen konstanten Drehmomentvektor zurück:
      tau_1 = 0.5 Nm am ersten Gelenk
      tau_2 = 0.2 Nm am zweiten Gelenk
    """
    if t is None:
        t = np.linspace(0, 1, N+1)  # Zeitgitter von 0 bis 1 s, N+1 Punkte
    tau_1 = np.full(t.shape, 0.5)
    tau_2 = np.full(t.shape, 0.2)
    return np.array([tau_1, tau_2]).T

# ------------------------------
# 3. Matrizenfunktionen D, C und Gravitationsterm G
# ------------------------------
def D_matrix(q1, q2, params):
    m1, m2 = params['m1'], params['m2']
    l1, l2 = params['l1'], params['l2']
    # Schwerpunkte: in der Mitte der Stäbe
    a1 = l1 / 2.0
    a2 = l2 / 2.0
    # Trägheitsmomente für homogene Stäbe (um den Schwerpunkt)
    I1 = m1 * l1**2 / 12.0
    I2 = m2 * l2**2 / 12.0

    D11 = m1*(a1**2) + I1 + m2*(l1**2 + a2**2 + 2 * l1 * a2 * np.cos(q2)) + I2
    D12 = m2*(a2**2 + l1 * a2 * np.cos(q2)) + I2
    D21 = D12
    D22 = m2*(a2**2) + I2
    return np.array([[D11, D12],
                     [D21, D22]])

def C_matrix(q1, q2, dq1, dq2, params):
    m2 = params['m2']
    l1, l2 = params['l1'], params['l2']
    lc2 = l2 / 2.0
    h = -m2 * l1 * lc2 * np.sin(q2)
    C11 = h * dq2
    C12 = h * (dq1 + dq2)
    C21 = -h * dq1
    C22 = 0.0
    return np.array([[C11, C12],
                     [C21, C22]])

def G_vector(q1, q2, params):
    m1, m2 = params['m1'], params['m2']
    l1, l2 = params['l1'], params['l2']
    g = params['g']
    lc1 = l1 / 2.0
    lc2 = l2 / 2.0
    # Klassische Formulierung mit cos:
    G1 = m1 * g * lc1 * np.cos(q1) + m2 * g * (l1 * np.cos(q1) + lc2 * np.cos(q1 + q2))
    G2 = m2 * g * lc2 * np.cos(q1 + q2)
    return np.array([G1, G2])

# ------------------------------
# 4. Zustandsableitung und Euler-Integration mittels For-Schleife
# ------------------------------
def state_derivative(state, t, params, current_tau):
    """
    Berechnet die Ableitung des Zustandsvektors [q1, q2, dq1, dq2]
    für einen gegebenen Zeitpunkt t unter Verwendung des aktuellen Drehmoments.
    """
    q1, q2, dq1, dq2 = state
    D = D_matrix(q1, q2, params)
    C = C_matrix(q1, q2, dq1, dq2, params)
    G = G_vector(q1, q2, params)
    dq_vec = np.array([dq1, dq2])
    # Dynamikgleichung: D * ddq = tau - C*dq - G
    ddq = np.linalg.solve(D, current_tau - C.dot(dq_vec) - G)
    return np.array([dq1, dq2, ddq[0], ddq[1]])

#---------------------------------------------------------------------------------------------

def simulate_robot(params, T=1.0, dt=0.001, N=1000):
    
    # Redundant T = dt * N = 1 
    
    t_values = np.linspace(0, T, N+1)
    state = np.zeros((N+1, 4))
    # Anfangszustand wählen (z.B. vertikaler Gleichgewichtszustand)
    state[0] = [np.pi/2, 0.0, 0.0, 0.0]
    
    # Erzeugen des konstanten Drehmomentverlaufs über das Zeitgitter:
    torque_array = tau(N, t_values)  # tau_array hat die Form (N+1, 2)

    for i in range(N):
        current_t = t_values[i]
        current_tau = torque_array[i]
        deriv = state_derivative(state[i], current_t, params, current_tau)
        state[i+1] = state[i] + dt * deriv

    return t_values, state

# ------------------------------
# 5. Animation zur Visualisierung der Roboterbewegung
# ------------------------------
def animate_robot(t_values, state, params):

    l1, l2 = params['l1'], params['l2']

    def forward_kinematics(q1, q2):
        x1 = l1 * np.cos(q1)
        y1 = l1 * np.sin(q1)
        x2 = x1 + l2 * np.cos(q1 + q2)
        y2 = y1 + l2 * np.sin(q1 + q2)
        return (0, 0), (x1, y1), (x2, y2)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(custom_background_dark)
    ax.set_facecolor(custom_background_dark)
    ax.set_xlim(- (l1 + l2) - 0.2, (l1 + l2) + 0.2)
    ax.set_ylim(- (l1 + l2) - 0.2, (l1 + l2) + 0.2)
    ax.set_aspect('equal')
    ax.grid(True, color=custom_background_light, linestyle='--', alpha=0.5)
    ax.tick_params(colors=custom_background_light)

    # Zwei separate Linien: Link 1 (Basis -> Gelenk) und Link 2 (Gelenk -> Endeffektor)
    line1, = ax.plot([], [], 'o-', lw=4, color=custom_green)
    line2, = ax.plot([], [], 'o-', lw=4, color=custom_red)

    def update(frame):
        q1, q2 = state[frame, 0], state[frame, 1]
        base, joint, end = forward_kinematics(q1, q2)
        line1.set_data([base[0], joint[0]], [base[1], joint[1]])
        line2.set_data([joint[0], end[0]], [joint[1], end[1]])
        ax.set_title(f"t = {t_values[frame]:.3f} s", color=custom_background_light)
        return line1, line2

    anim = FuncAnimation(fig, update, frames=len(t_values), interval=10, blit=True)
    plt.show()


# ------------------------------
# Hauptprogramm
# ------------------------------
if __name__ == '__main__':
    T = 10.0     # Gesamtdauer der Simulation in Sekunden
    dt = 0.0001  # Zeitschritt in Sekunden
    N = 10000    # Anzahl der diskretisierten Schritte

    t_values, state = simulate_robot(params, T, dt, N)
    
    # Zunächst können Sie die Gelenkwinkel als Plot ansehen:

    plt.figure(figsize=(10, 5), facecolor=custom_background_dark)
    ax = plt.gca()
    ax.set_facecolor(custom_background_dark)

    plt.plot(t_values, state[:, 0], label='q1 (rad)', color=custom_green, linewidth=2)
    plt.plot(t_values, state[:, 1], label='q2 (rad)', color=custom_red, linewidth=2)
    plt.xlabel('Zeit (s)', color=custom_background_light)
    plt.ylabel('Gelenkwinkel (rad)', color=custom_background_light)
    plt.title('Entwicklung der Gelenkwinkel über die Zeit', color=custom_background_light)
    plt.legend(facecolor=custom_background_dark, edgecolor=custom_background_light)
    plt.grid(True, color=custom_background_light, linestyle='--', alpha=0.5)
    ax.tick_params(colors=custom_background_light)
    plt.show()


    # Anschließend die Animation:
    animate_robot(t_values, state, params)
