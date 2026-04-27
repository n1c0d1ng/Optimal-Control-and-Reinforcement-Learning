import numpy as np
from NN import Neural_Network_One_Layer
from ControlProblem import OCP

batch_size = 2
tau = 0.001    # weiche Update-Rate
learning_rate_actor = 1e-4
learning_rate_critic = 1e-3

def train_DDPG(Environment = OCP(100),num_episodes = 4, learning_rate = 0.0001, Capacity = 10000):
    # Initialisieren der Netzwerke
    u = Neural_Network_One_Layer(n=5, input_dim=2, output_dim=1)          # Actor
    Q = Neural_Network_One_Layer(n=10, input_dim=3, output_dim=1)         # Critic
    u_target = Neural_Network_One_Layer(n=5, input_dim=2, output_dim=1)   # Target Actor
    Q_target = Neural_Network_One_Layer(n=10, input_dim=3, output_dim=1)  # Target Critic

    # Selbe Initialisierung
    u_target.set_parameters(u.get_parameters())
    Q_target.set_parameters(Q.get_parameters())

    # Anlegen des Replay Buffers
    Buffer = np.zeros((Capacity, 7)) 
    index_counter = 0
    numbers_of_samples = 0

    for episode in range(num_episodes):
        x = 0.0
        t = 0.0
        exploration = max(0.1, 1.0 - episode/num_episodes)  

        # Generiere Trajektorien und speichern im Buffer
        for step in range(Environment.N):

            # Frage wie genau wie liest das NN den Input ein ?

            action = u.ForwardPass(np.array([t,x]))[0] + np.random.normal(0,exploration)
            x_next = Environment.get_next_state(x,action)
            reward = Environment.direct_reward(x,action)

            if step == Environment.N - 1: 
                reward += Environment.terminal_reward(x_next)
                done = 1
            else: done = 0

            t_next = t + Environment.h

            # Füllen des Buffers: (Cx7)
            # (s_t= [t_i X_i] , a_t , r_t , s_t+1 = [t_i+1 , X_i+1] , done)
            Buffer[index_counter] = np.array([t,x,action, reward,t_next, x_next, done])


            if index_counter == Capacity -1:
                index_counter = 0
            else: 
                index_counter += 1
                numbers_of_samples += 1

            # Ziehen unserer y_i aus dem Buffer:
            if numbers_of_samples >= batch_size:

                # Zufällige Indizes für den Batch ziehen
                indices = np.random.choice(numbers_of_samples, batch_size, replace=False)
                batch = Buffer[indices]  # (batch_size, 7)

                # Daten extrahieren
                s_batch = batch[:, 0:2]         # (t, x)
                a_batch = batch[:, 2]
                r_batch = batch[:, 3]
                s_next_batch = batch[:, 4:6]    # (t_next, x_next)
                done_batch = batch[:, 6]        # done 

                y = np.zeros(batch_size)
                Q_current = np.zeros(batch_size)
                grad_phi = np.zeros(Q.parameter_dim)

                # Berechnung y_i = r_i + Q(s_i+1,u(s_i+1)) unter phi' und theta'
                for i in range(batch_size):
                    s_next = np.array([s_next_batch[i, 0], s_next_batch[i, 1]])
                    a_next = u_target.ForwardPass(s_next)[0]   
                    q_next = Q_target.ForwardPass(np.array([s_next[0], s_next[1], a_next]))[0]
                    y[i] = r_batch[i] + (1 - done_batch[i]) * q_next

                    s = np.array([s_batch[i, 0], s_batch[i, 1]])            # s_i = (t_i,X_i)
                    q = Q.ForwardPass(np.array([s[0], s[1], a_batch[i]]))[0] # Q_phi(s_i,a_i)
                    Q_current[i] = q

                    # Critic-Update: L(psi) = 1/B sum_i (y_i-Q(s_i,a_i))^2 
                    # -> L'= 2/B sum_i (y_i-Q(s_i,a_i)) * grad_Q
                    delta = 2 * (Q_current[i] - y[i]) / batch_size   
                    grad_Q = Q.Backpropagation(np.array([s_batch[i,0], s_batch[i,1], a_batch[i]]))
                    grad_phi += delta * grad_Q

                # Aktualisierung der Parameter von Q_phi
                new_phi = Q.get_parameters() - learning_rate_critic * grad_phi
                Q.set_parameters(new_phi)

                grad_theta = np.zeros(u.parameter_dim)
                for i in range(batch_size):
                    s = np.array([s_batch[i,0], s_batch[i,1]])
                    a_pred = u.ForwardPass(s)[0]   # aktuelle Aktion laut Actor
                    # Gradient von Q nach a
                    grad_input = Q.sensitivities(np.array([s[0], s[1], a_pred]))
                    grad_a = grad_input[2]          # dritte Komponente
                    # Gradient von u nach theta
                    grad_u = u.Backpropagation(s)
                    grad_theta += grad_a * grad_u
                grad_theta /= batch_size

                new_theta = u.get_parameters() + learning_rate_actor * grad_theta
                u.set_parameters(new_theta)

                new_phi_target = tau * Q.get_parameters() + (1 - tau) * Q_target.get_parameters()
                Q_target.set_parameters(new_phi_target)
                new_theta_target = tau * u.get_parameters() + (1 - tau) * u_target.get_parameters()
                u_target.set_parameters(new_theta_target)

            # Update der Simulation
            t = t_next
            x = x_next

            # # Update Critic: 
            # Q(s_i,a_i) = r_i + Q(s_i+1,u(s_i+1))
            # Hier ziehen Batch aus Buffer und dann y = r +  Q(s,u(s')) und Q = Q(s,a) 
            # Berechne Loss und minimiere L(psi) = sum (y-Q_psi)^2 -> min
            # also y sind die theoretischen und Q die empirischen

            # Update Actor

    print("Succesful")

# Aufruf des Programms zu testzwecken
if __name__ == "__main__":
    train_DDPG(Environment=OCP(100))