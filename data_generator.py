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
            x0 = np.random.uniform(-1, 1, self.system.num_states)
            u = np.random.uniform(-1, 1, self.system.num_inputs)
            sample = self.system.step(num_steps=1, ts=ts, x0=x0, u=u)
            u_vector[i] = u
            y_vector[i] = sample
        
        generator = TimeseriesGenerator(u_vector, y_vector, length=delay, batch_size=batch_size)
        
        return generator
