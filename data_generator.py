import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator


class DynamicSystemDataGenerator:
    def __init__(self, system) -> None:
        self.system = system
        pass

    def get_data_generator(self, samples=1000, ts=0.01, delay=2, batch_size=1):
        """
        Data generator that produces training data from a dynamic system of equations
        :param samples (int):    Number of points for each state variable
        :param ts (float):       Sampling time
        :param delay (int):      Time delay (Lag time)
        :param batch_size (int): Batch size
        return: raw dataset
        """
        u_vector = np.zeros((samples, self.system.num_inputs))
        y_vector = np.zeros((samples, self.system.num_states))
        for i in range(samples):
            x0 = np.random.uniform(-10, 10, self.system.num_states)
            u = np.random.uniform(-10, 10, self.system.num_inputs)
            sample = self.system.step(num_steps=1, ts=ts, x0=x0, u=u)
            u_vector[i] = u
            y_vector[i] = sample

        # Autoregressive data
        u_vector, y_vector = self._construct_regression_vector(u_vector, y_vector, delay=delay)
        
        generator = TimeseriesGenerator(u_vector, y_vector, length=delay, batch_size=batch_size)
        
        return generator

    def _construct_regression_vector(self, u_vector, y_vector, delay):
        """
        Creates lagged versions of u and y samples given the delay value
        :param u_vector (FROM get_data_generator method)
        :param y_vector (FROM get_data_generator method)
        :param delay (int) same for u and y. Independent TODO
        """
        new_u = u_vector.copy()
        _aux_u = u_vector.copy()
        new_y = y_vector.copy()
        _aux_y = y_vector.copy()
        for _ in range(delay):
            # Roll u vector 
            _aux_u = np.roll(_aux_u, 1, axis=0)
            # Fill with zeroes or Initial conditions TODO
            _aux_u[0] = 0
            new_u = np.hstack((new_u, _aux_u))

        for _ in range(delay):
            # Roll Y vector 
            _aux_y = np.roll(_aux_y, 1, axis=0)
            # Fill with zeroes or Initial conditions TODO
            _aux_y[0] = 0
            new_u = np.hstack((new_u, _aux_y))
        return new_u, new_y
