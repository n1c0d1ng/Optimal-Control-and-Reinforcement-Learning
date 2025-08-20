import numpy as np
import matplotlib.pyplot as plt

# Parameter
Parameters = {
    'm1': 1.0,       # Masse von Link 1 in kg
    'm2': 1.0,       # Masse von Link 2 in kg
    'l1': 1.0,       # Länge von Link 1 in m
    'l2': 1.0,       # Länge von Link 2 in m
    'g': 9.81        # Erdbeschleunigung in m/s^2
}

# Definition von D
def D_matrix(theta, Parameters):

    # Auslesen der Parameter
    m1 = Parameters['m1']
    m2 = Parameters['m2']
    l1 = Parameters['l1']
    l2 = Parameters['l2']

    # Berechnung Schwerpunkt der Stäbe
    a1 = l1 / 2.0
    a2 = l2 / 2.0

    # Definition Moment 
    I1 = m1 * l1**2 / 12.0
    I2 = m2 * l2**2 / 12.0

    D11 = m1*(a1**2) + I1 + m2*(l1**2 + a2**2 + 2 * l1 * a2 * np.cos(theta[1])) + I2
    D12 = m2*(a2**2 + l1 * a2 * np.cos(theta[1])) + I2
    D21 = D12
    D22 = m2*(a2**2) + I2

    return np.array([[D11, D12],[D21, D22]])


# Definition von C
def C_matrix(theta,Dtheta, Parameters):

    # Auslesen der Parameter
    m1 = Parameters['m1']
    m2 = Parameters['m2']
    l1 = Parameters['l1']
    l2 = Parameters['l2']

    # Berechnung Schwerpunkt der Stäbe
    a1 = l1 / 2.0
    a2 = l2 / 2.0

    C11 = -2*m2*l1*a2*np.sin(theta[1])*Dtheta[1]
    C12 = -m2*l1*a2*np.sin(theta[1])*Dtheta[1]
    C21 = m2*l1*a2*np.sin(theta[1])*Dtheta[0]
    C22 = 0

    return np.array([[C11, C12],[C21, C22]])

# Definition von G
def G_matrix(theta, Parameters):

        # Auslesen der Parameter
    m1 = Parameters['m1']
    m2 = Parameters['m2']
    l1 = Parameters['l1']
    l2 = Parameters['l2']
    g = Parameters['g']

    # Berechnung Schwerpunkt der Stäbe
    a1 = l1 / 2.0
    a2 = l2 / 2.0

    G1 = m1*g*a1*np.cos(theta[0])+m2*g*(l1*np.cos(theta[0])+a2*np.cos(theta[0]+theta[1]))
    G2 = m2*g*(l1*np.cos(theta[0])+a2*np.cos(theta[0]+theta[1]))

    return np.array([G1,G2])


if __name__ == '__main__':

    # Diskretisierung h*n = T = [0,1]
    T = 1
    h = 0.1 
    n = T/h

    # Teststeuerung 
    # --------------------------------------------------------------
    def Control_funtion(Number_of_Steps):
        tau1 = 0.2*np.ones(int(Number_of_Steps)+1)
        tau2 = 0.1*np.ones(int(Number_of_Steps)+1)
        return np.array([tau1,tau2]).T
    # -------------------------------------------------------------
    
    # Testweise ein Integrationsschritt x_0 -> x_1 
    # x = [theta , Dtheta]
    #-------------------------------------------------------------
    theta = np.zeros((int(n),2))
    Dtheta = np.zeros((int(n),2))

    theta[0] = np.array([np.pi/4, 0])
    Dtheta[0] = np.array([0, 0])

    # Update Schritt
    #---------------------------
    

    #--------------------------


    u = Control_funtion(n)
    D = D_matrix(np.array([np.pi/4 ,np.pi/2]), Parameters)

    print(Control_funtion)
    print("Hello WOrlds")
