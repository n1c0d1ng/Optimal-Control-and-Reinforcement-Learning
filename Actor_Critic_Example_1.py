# Actor-Critic Implementierung für SCOP:
# min_pi int_0^1 (X_t + u_t^2) dt + X_1^2

from Stochastic_Environment import SCOP
import matplotlib.pyplot as plt
import numpy as np

Environment = SCOP(N=100)

