# Implementierung von Beispiel 3.1
# min int_0^1 x+u^2 dt x'=x+u+1 x(0) = 0
#--------------------------------------------------------------------------------------------------
import Functions
import matplotlib.pyplot as plt
import numpy as np


# Meine Standardfarben für Darkmode
custom_background_dark = (53/255, 62/255, 69/255)  
custom_background_light = (255/255, 255/255, 255/255)
custom_red = (255/255, 94/255, 77/255)
custom_blau = (70/255, 130/255, 255/255)
custom_orange = (255/255,127/255,80/255)



def p_analytic(t):
    return -1 + np.exp(-t + 1)

def u_analytic(t):
    return 0.5 * (1 - np.exp(-t + 1))

def x_analytic(t):
    return -1.5 + (np.exp(-t + 1) / 4) + 1.5 * np.exp(t) - (np.exp(t + 1) / 4)

def plot_comparison(t1, x1, u1, t2, x2, u2):
    # Analytische Lösungen
    t_analytic = np.linspace(0, 1, 1000)
    x_ana = x_analytic(t_analytic)
    u_ana = u_analytic(t_analytic)

    plt.figure(figsize=(15, 5), facecolor= custom_background_dark)


    # Vergleich für x(t)
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(t_analytic, x_ana, label="Analytisch: x(t)", color=custom_red, linewidth=1)
    ax1.plot(t1, x1, label="Numerisch (N=100): x(t)", color=custom_blau, linestyle="-")
    ax1.plot(t2, x2, label="Numerisch (N=1000): x(t)", color=custom_orange, linestyle="-")
    #ax1.set_title("Zustand x(t)")
    #ax1.set_xlabel("t")
    #ax1.set_ylabel("x")

    # Anpassung der Achsenlinien
    ax1.spines['top'].set_visible(False)  # Obere Linie ausblenden
    ax1.spines['right'].set_visible(False)  # Rechte Linie ausblenden
    ax1.spines['bottom'].set_color(custom_background_light)  # Untere Linie 
    ax1.spines['left'].set_color(custom_background_light)  # Linke Linie 
    ax1.tick_params(axis='x', colors=custom_background_light)  # X-Ticks 
    ax1.tick_params(axis='y', colors=custom_background_light)  # Y-Ticks 

    # Vergleich für u(t)
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(t_analytic, u_ana, label="Analytisch: u(t)", color=custom_red, linewidth=1)
    ax2.plot(t1, u1, label="Numerisch (N=100): u(t)", color=custom_blau, linestyle="-")
    ax2.plot(t2, u2, label="Numerisch (N=1000): u(t)", color=custom_orange, linestyle="-")
    #ax2.set_title("Steuerung u(t)")
    #ax2.set_xlabel("t")
    #ax2.set_ylabel("u")

    # Anpassung der Achsenlinien
    ax2.spines['top'].set_visible(False)  # Obere Linie ausblenden
    ax2.spines['right'].set_visible(False)  # Rechte Linie ausblenden
    ax2.spines['bottom'].set_color(custom_background_light)  # Untere Linie 
    ax2.spines['left'].set_color(custom_background_light)  # Linke Linie 
    ax2.tick_params(axis='x', colors=custom_background_light)  # X-Ticks 
    ax2.tick_params(axis='y', colors=custom_background_light)  # Y-Ticks 

    ax1.set_facecolor(custom_background_dark)
    ax2.set_facecolor(custom_background_dark)

    # Layout anpassen und Plot anzeigen
    plt.tight_layout()
    plt.savefig("plot.pdf", format="pdf", bbox_inches="tight")
    plt.show()





# Berechnung der numerischen Lösungen
u_num, x_num, t = Functions.gradient_descent(100, 100, 0.00001)
u_num2, x_num2, t2 = Functions.gradient_descent(100, 500, 0.00001)


# Vergleich in einem einzigen Plot
plot_comparison(t, x_num, u_num, t2, x_num2, u_num2)
