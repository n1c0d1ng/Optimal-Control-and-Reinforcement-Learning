# Grid: 3x3 mit folgender Struktur
# R |   | Y
# --|---|xxx
# G |   | B
# --|---|---
#   X   |  

grid_size = (3, 3)  # 3 Zeilen, 3 Spalten

# Plätze mit Namen und Koordinaten
places = {
    'R': (0, 0),
    'G': (1, 0),
    'Y': (0, 2),
    'B': (1, 2)
}

# Mauern als Liste von Paaren (von Feld zu Feld)
walls = [
    ((0, 2), (1, 2)),  # Mauer zwischen (0,2) und (1,2)
    ((2, 0), (2, 1))   # Mauer zwischen (2,0) und (2,1)
]

# Prüfe ob zwischen den Feldern eine Mauer ist
# Gib True zurück wenn Bewegung nicht möglich ist
def is_blocked(from_pos, to_pos, walls):
    return (from_pos, to_pos) in walls or (to_pos, from_pos) in walls

passenger_locations = list(places.keys()) + ['IN_TAXI']
destinations = list(places.keys())

# Aufbau der Zustände als Liste von Tupeln
states = []
for taxi_x in range(grid_size[0]):
    for taxi_y in range(grid_size[1]):
        for passenger in passenger_locations:
            for destination in destinations:
                # Hier Tupel für jeden Zustand erstellen
                states.append((taxi_x, taxi_y, passenger, destination))

# Actionen definieren
actions = ['south', 'north', 'east', 'west', 'pickup', 'dropoff']

# Definition der Bewegung 
move = {
    'south': (1, 0),
    'north': (-1, 0),
    'east': (0, 1),
    'west': (0, -1)
}

# Dictionary für die Dynamik: transitions[state][action] = (next_state, reward, done)
#--------------------------------------------------------------------------------
transitions = {}
for state in states:
    transitions[state] = {}
    taxi_x, taxi_y, passenger_loc, destination = state
    for action in actions:
        # Initialwerte
        next_state = state
        reward = -1  # Standard-Strafpunkt für jeden Schritt
        done = False

        # Prüfe ob wir einen Bewegungsbefehl machen
        if action in move:
            # Bewegung
            dx, dy = move[action]
            new_x, new_y = taxi_x + dx, taxi_y + dy
            # Prüfe ob neue Position innerhalb des 3x3 Grids liegt
            if 0 <= new_x < grid_size[0] and 0 <= new_y < grid_size[1]:
                # Prüfe ob Bewegung durch Mauer blockiert ist
                if is_blocked((taxi_x, taxi_y), (new_x, new_y), walls) == True:
                    # Bewegung durch Mauer blockiert
                    next_state = (taxi_x, taxi_y, passenger_loc, destination)
                else:
                    # Bewegung erlaubt
                    next_state = (new_x, new_y, passenger_loc, destination)
            else:
                # Bewegung außerhalb des Grids
                next_state = (taxi_x, taxi_y, passenger_loc, destination)
        elif action == 'pickup':
            # Passagier aufnehmen
            if (passenger_loc != 'IN_TAXI' \
                and (taxi_x, taxi_y) == places[passenger_loc]):
                reward = 10  # Belohnung für erfolgreichen Pickup
                next_state = (taxi_x, taxi_y, 'IN_TAXI', destination)
            else:
                reward = -10  # Strafpunkt für fehlerhaften Pickup
        elif action == 'dropoff':
            # Passagier absetzen
            if (passenger_loc == 'IN_TAXI' \
                and (taxi_x, taxi_y) == places[destination]):
                next_state = (taxi_x, taxi_y, destination, destination)
                reward = 20  # Belohnung für erfolgreichen Dropoff
                done = True
            else:
                reward = -10  # Strafpunkt für fehlerhaften Dropoff
        transitions[state][action] = (next_state, reward, done)


#--------------------------------------------------------------------------------
# V_k+1 = max_a E[ R(a,s,s') + γ * V_k(s') ]
#       = max_a sum_s' T_a(s,s') [ R(a,s,s') + γ * V_k(s') ]
#       aber T_a(s,s') = 0 oder 1 
#--------------------------------------------------------------------------------
def value_iteration(states, transitions, gamma=0.9, epsilon=1e-6, max_iterations=1000):
    # Initialisiere V(s) = 0 für alle 180 Zustände als Dictionary
    V = {state: 0.0 for state in states}
    
    for iteration in range(max_iterations):
        V_new = {}
        max_delta = 0
        
        for state in states:
            # Überspringe Terminalzustände (wo Passagier bereits am Ziel ist)
            passenger_loc, destination = state[2], state[3] 
            if passenger_loc == destination and passenger_loc != 'IN_TAXI':
                V_new[state] = 0
                # Ausstieg aus der Schleife
                continue
                
            best_value = -float('inf')
            # Hier Maximierung über Aktionen
            for action in actions:
                next_state, reward, done = transitions[state][action]
                
                # Wenn Terminalzustand erreicht, future_value = 0
                if done == True:
                    future_value = 0
                else:
                    # Im ersten Aufruf immer 0 da V(s) = 0 initialisiert
                    future_value = V[next_state]
                
                # Q(s,a) = R(s,a,s') + γ * V(s')
                action_value = reward + gamma * future_value
                
                if action_value > best_value:
                    best_value = action_value
            
            V_new[state] = best_value
            max_delta = max(max_delta, abs(V_new[state] - V[state]))
        V = V_new
        
        # Abbruch Bedingung formulieren
        if max_delta < epsilon:
            print(f"Konvergenz nach {iteration + 1} Iterationen")
            break
    return V

# Nach der Value Iteration: Policy Extraction mit Q
#--------------------------------------------------------------------------------
def extract_policy(V, transitions, gamma=0.9):
    Q_star = {}
    policy = {}
    
    for state in states:
        Q_star[state] = {}
        best_value = -float('inf')
        best_action = None
        
        for action in actions:
            next_state, reward, done = transitions[state][action]
            if done == True:
                future_value = 0
            else:
                future_value = V[next_state]

            q_value = reward + gamma * future_value
            Q_star[state][action] = q_value
            
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        policy[state] = best_action
    return Q_star, policy

# 1. Value Iteration durchführen
#--------------------------------------------------------------------------------
print("Starte Value Iteration...")
V_star = value_iteration(states, transitions)

# 2. Policy und Q* extrahieren
#--------------------------------------------------------------------------------
print("Extrahiere optimale Policy und Q-Funktion...")
Q_star, policy_star = extract_policy(V_star, transitions)

# 3. Ergebnisse anzeigen
#--------------------------------------------------------------------------------
sample_state = (1, 2, 'B', 'Y')
print(f"\nTestzustand: Taxi bei (1,2), Passagier bei B, Ziel bei Y")
V = value_iteration(states, transitions)
Q_star, policy = extract_policy(V, transitions)

Trajektorie = []
while True:
    action = policy[sample_state]
    next_state, reward, done = transitions[sample_state][action]
    Trajektorie.append((sample_state, action, reward, next_state))
    print(f"Zustand: {sample_state}, Aktion: {action}, Belohnung: {reward}")
    sample_state = next_state
    if done == True:
        break