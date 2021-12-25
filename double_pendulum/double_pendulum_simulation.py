from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
import time

from myterial.utils import make_palette
from myterial import grey_darker, pink, blue_light

class Pendulum:
    G: float = 9.8  # acceleration due to gravity, in m/s^2
    L1: float = 1.0  # length of pendulum 1 in m
    L2: float = 1.0  # length of pendulum 2 in m
    M1: float = 1.0  # mass of pendulum 1 in kg
    M2: float = 1.0  # mass of pendulum 2 in kg

    def __init__(self,
        th1: float, w1:float, th2:float, w2:float
    ):
        '''
            th1 and th2 are the initial angles (degrees)
            w1 and w2 are the initial angular velocities (degrees per second)
        '''

        # create state
        state = np.radians([th1, w1, th2, w2])

        # integrate dynamics
        dynamics = integrate.odeint(self.derivatives, state, T)

        # get points position
        self.x1 = self.L1*sin(dynamics[:, 0])
        self.y1 = -self.L1*cos(dynamics[:, 0])

        self.x2 = self.L2*sin(dynamics[:, 2]) + self.x1
        self.y2 = -self.L2*cos(dynamics[:, 2]) + self.y1


    def derivatives(self, state, *args):
        '''
            Derivative of the dynamics
        '''
        dydx = np.zeros_like(state)
        dydx[0] = state[1]

        delta = state[2] - state[0]
        den1 = (self.M1+self.M2) * self.L1 - self.M2 * self.L1 * cos(delta) * cos(delta)
        dydx[1] = ((self.M2 * self.L1 * state[1] * state[1] * sin(delta) * cos(delta)
                    + self.M2 * self.G * sin(state[2]) * cos(delta)
                    + self.M2 * self.L2 * state[3] * state[3] * sin(delta)
                    - (self.M1+self.M2) * self.G * sin(state[0]))
                / den1)

        dydx[2] = state[3]

        den2 = (self.L2/self.L1) * den1
        dydx[3] = ((- self.M2 * self.L2 * state[3] * state[3] * sin(delta) * cos(delta)
                    + (self.M1+self.M2) * self.G * sin(state[0]) * cos(delta)
                    - (self.M1+self.M2) * self.L1 * state[1] * state[1] * sin(delta)
                    - (self.M1+self.M2) * self.G * sin(state[2]))
                / den2)

        return dydx


# create a time array from 0..100 sampled at 0.05 second steps
n_seconds = 100
dt = 0.001
T = np.arange(0, n_seconds, dt)

# params
th1 = 120.0
w1 = 0.0
th2 = -10.0
w2 = 0.0


# create a bunch of pendulum
start = time.time()
pend = Pendulum(th1, w1, th2, w2)
end = time.time()
print(f'Simulation took: {end-start} for {len(pend.x1)} simulation steps')

