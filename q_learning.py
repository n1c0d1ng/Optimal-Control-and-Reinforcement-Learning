# Implementierung von Q Learning Algorithmus
import grid_world 
import numpy as np

env = grid_world.Grid_World(width=3, height=3)

def q_learning(env, alpha=0.001, gamma=0.9, epsilon=0.15, max_iterations=6970):

    # Initialisierung der Q-Tabelle
    Q = {}
    for state in env.states:
        Q[state] = {}
        for action in env.actions:
            Q[state][action] = 0.0

    num_actions = len(env.actions)
    
    for iteration in range(max_iterations):
        state = env.reset_random()
        done = False
        
        while done == False:
            # epsilon-greedy Aktion:
            # P(pi) = epsilon P(pi*) = 1 - epsilon
            if np.random.random() < epsilon:
                action = env.actions[np.random.randint(num_actions)]
            else:
                action = max(Q[state], key=Q[state].get)

            # Interaktion mit der Umgebung als BlackBox
            next_state, reward, done = env.get_next_state(state, action)
            
            # Q-Learning Update mit max_a Q(s',a)
            Q[state][action] += alpha * (
                reward + gamma * max(Q[next_state].values()) - Q[state][action]
            )
            state = next_state
    return Q

def extract_optimal_policy(Q, env):
    policy = {}
    for state in env.states:
        # Wähle die Aktion mit dem höchsten Q-Wert für jeden Zustand
        best_action = max(Q[state], key=Q[state].get)
        policy[state] = best_action
    return policy

# Beispielaufruf
#---------------------------------------------------------------------------------
Trajektorie = []
sample_state = (2, 1, 'B', 'Y')
Q = q_learning(env)
policy = extract_optimal_policy(Q, env)
while True:
    action = policy[sample_state]
    next_state, reward, done = env.get_next_state(sample_state, action)
    Trajektorie.append((sample_state, action, reward, next_state))
    print(f"Zustand: {sample_state}, Aktion: {action}, Belohnung: {reward}")
    sample_state = next_state
    if done == True:
        break