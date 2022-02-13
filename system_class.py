from re import S
import numpy as np
from scipy.integrate import odeint

class System:
    def __init__(self) -> None:
        self.num_states = None
        self.num_inputs = None
        pass

    @staticmethod
    def _dynamics(x, t, u):
        return

    def simulation(self, simulation_time=1.0, ts=0.01):
        return

    def step(self, num_steps=1, ts=0.01, x0=None, u=1.0):
        """
        :param num_steps (int) Number of future steps to simulate
        :param ts (float) Sample time 
        :param x0 (ndarray) Initial conditions
        :param u (ndarray) Input vector
        :return (ndarray) Simulation-steps results
        """
        if x0 is None:
            x0 = np.zeros(self.num_states)
        
        complete_result = np.zeros((num_steps, self.num_states))
        t = 0
        # Simulate steps
        for _step in range(num_steps):
            result_states = odeint(self._dynamics, y0=x0, t=[t, t + ts], args=(u,))
            t = ts
            x0 = result_states[-1].copy()
            complete_result[_step] = result_states[-1]

        return complete_result