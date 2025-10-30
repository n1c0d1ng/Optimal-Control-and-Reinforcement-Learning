# Grid: 3x3 mit folgender Struktur
# R |   | Y
# --|---|xxx
# G |   | B
# --|---|---
#   X   |  
import random
class Grid_World:   
    def __init__(self, width=3, height=3, places=None, walls=None):
        self.width = width
        self.height = height

        if places is None:
            self.places = {
                'R': (0, 0),
                'G': (1, 0),
                'Y': (0, 2),
                'B': (1, 2)
            }
        else:
            self.places = places

        if walls is None:
            self.walls = [
                    ((0, 2), (1, 2)),  # Mauer zwischen (0,2) und (1,2)
                    ((2, 0), (2, 1))   # Mauer zwischen (2,0) und (2,1)
            ]
        else:
            self.walls = walls

        # Actionen definieren
        self.actions = ['south', 'north', 'east', 'west', 'pickup', 'dropoff']

        # Definition der Bewegung
        self.move = {
            'south': (1, 0),
            'north': (-1, 0),
            'east': (0, 1),
            'west': (0, -1)
        }
        self.states = self.build_states()  
        self.transitions = self.build_transitions()

    def build_states(self):
        states = []
        passenger_locations = list(self.places.keys()) + ['IN_TAXI']
        destinations = list(self.places.keys())
        for taxi_x in range(self.width):
            for taxi_y in range(self.height):
                for passenger in passenger_locations:
                    for destination in destinations:
                        # Hier Tupel für jeden Zustand erstellen
                        states.append((taxi_x, taxi_y, passenger, destination))
        return states

    def is_blocked(self, from_pos, to_pos):
        return (from_pos, to_pos) in self.walls or (to_pos, from_pos) in self.walls
    
    def build_transitions(self):
        transitions = {}
        for state in self.states:
            transitions[state] = {}
            taxi_x, taxi_y, passenger_loc, destination = state
            for action in self.actions:
                # Initialwerte
                next_state = state
                reward = -1  # Standard-Strafpunkt für jeden Schritt
                done = False

                # Prüfe ob wir einen Bewegungsbefehl machen
                if action in self.move:
                    # Bewegung
                    dx, dy = self.move[action]
                    new_x, new_y = taxi_x + dx, taxi_y + dy
                    # Prüfe ob neue Position innerhalb des 3x3 Grids liegt
                    if 0 <= new_x < self.width and 0 <= new_y < self.height:
                        # Prüfe ob Bewegung durch Mauer blockiert ist
                        if self.is_blocked((taxi_x, taxi_y), (new_x, new_y)):
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
                        and (taxi_x, taxi_y) == self.places[passenger_loc]):
                        reward = 10  # Belohnung für erfolgreichen Pickup
                        next_state = (taxi_x, taxi_y, 'IN_TAXI', destination)
                    else:
                        reward = -10  # Strafpunkt für fehlerhaften Pickup
                elif action == 'dropoff':
                    # Passagier absetzen
                    if (passenger_loc == 'IN_TAXI' \
                        and (taxi_x, taxi_y) == self.places[destination]):
                        next_state = (taxi_x, taxi_y, destination, destination)
                        reward = 20  # Belohnung für erfolgreichen Dropoff
                        done = True
                    else:
                        reward = -10  # Strafpunkt für fehlerhaften Dropoff
                transitions[state][action] = (next_state, reward, done)
        return transitions

    def get_next_state(self, state, action):
        return self.transitions[state][action]
    
    def reset_random(self):
        return random.choice(self.states)