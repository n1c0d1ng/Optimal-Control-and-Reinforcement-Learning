#--------------------------------------------------------------------------------
# V_k+1 = max_a E[ R(a,s,s') + γ * V_k(s') ]
#       = max_a sum_s' T_a(s,s') [ R(a,s,s') + γ * V_k(s') ]
#       aber T_a(s,s') = 0 oder 1 
#--------------------------------------------------------------------------------

import grid_world 
def value_iteration(states, actions, transitions, gamma=0.9, epsilon=1e-6, max_iterations=1000):
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
def extract_policy(states, actions, V, transitions, gamma=0.9):
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

# 0. Grid World Environment initialisieren
#--------------------------------------------------------------------------------
Env = grid_world.Grid_World(width=3,height=3)

# 1. Value Iteration durchführen
#--------------------------------------------------------------------------------
print("Starte Value Iteration...")
V_star = value_iteration(Env.states, Env.actions, Env.transitions)

# 2. Policy und Q* extrahieren
#--------------------------------------------------------------------------------
print("Extrahiere optimale Policy und Q-Funktion...")
Q_star, policy_star = extract_policy(Env.states, Env.actions, V_star, Env.transitions)

# 3. Ergebnisse anzeigen
#--------------------------------------------------------------------------------
sample_state = (2, 1, 'B', 'Y')
#print(f"\nTestzustand: Taxi bei (1,2), Passagier bei B, Ziel bei Y")
V = value_iteration(Env.states, Env.actions, Env.transitions)
Q_star, policy = extract_policy(Env.states, Env.actions, V, Env.transitions)

Trajektorie = []
while True:
    action = policy[sample_state]
    next_state, reward, done = Env.get_next_state(sample_state, action)
    Trajektorie.append((sample_state, action, reward, next_state))
    print(f"Zustand: {sample_state}, Aktion: {action}, Belohnung: {reward}")
    sample_state = next_state
    if done == True:
        break