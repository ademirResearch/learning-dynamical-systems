import numpy as np
from keras.preprocessing.sequence import TimeseriesGenerator
# from rsa import sign
import tensorflow as tf
import matplotlib.pyplot as plt


class DynamicSystemDataGenerator:
    def __init__(self, system) -> None:
        self.system = system
        self.scale_parameters = None
        pass

    def _get_psrb_signal(self, samples=1000, sequence_limits=(1, 10), magnitude_limits=(-1, 1)):
        """Computes Pseudorandom binary sequence signal
        Args:
            samples (int, optional): Number of total samples Defaults to 1000.
            sequence_limits (tuple, optional): Range of possible sequences lengths. Defaults to (1, 10).
            magnitude_limits (tuple, optional): Range of possible magnitudes. Defaults to (-1, 1).
        """
        i = 0
        psrb_signal = np.zeros(1)
        while i < samples: 
            sequence_length = np.random.randint(sequence_limits[0], sequence_limits[1])
            signal = np.zeros(sequence_length)
            magnitude = np.random.uniform(magnitude_limits[0], magnitude_limits[1])
            for j in range(sequence_length):
                signal[j] = 1 * magnitude
                i += 1
            psrb_signal = np.concatenate((psrb_signal, signal), axis=0)
        
        # Delete first zero used for concatenation
        psrb_signal = np.delete(psrb_signal, 0, axis=0)
        return psrb_signal[:samples]

    def get_data_generator(self, experiments=2, samples=1000, ts=0.01, delay=2, batch_size=8, state="all"):
        """
        Data generator that produces training data from a dynamic system of equations
        :param samples (int):    Number of points for each state variable
        :param ts (float):       Sampling time
        :param delay (int):      Time delay (Lag time)
        :param batch_size (int): Batch size
        :param state (string):   All states or specific
        return: raw dataset
        """
        # _random_sequence_length = np.random.randint(100, 1000)
        u_vector = np.zeros((samples, self.system.num_inputs))
        y_vector = np.zeros((samples, self.system.num_states))
        # for i in range(int(samples/_random_sequence_length)):
        #     x0 = np.random.uniform(-1, 1, self.system.num_states)
        #     u = np.random.uniform(-1, 1, self.system.num_inputs)
        #     sample = self.system.step(num_steps=_random_sequence_length, ts=ts, x0=x0, u=u)
        #     for _ in range(_random_sequence_length):
        #         u_vector[i*_random_sequence_length + _] = u
        #         y_vector[i*_random_sequence_length + _] = sample[_]

        # Remove unwanted states if any
        if state == "all":
            pass
        else:
            state = int(state)
            y_vector = y_vector[:, state].reshape(-1, 1)

        # Conduct experiments
        dummy_u = np.zeros((1, self.system.num_states))  # Column 
        dummy_y = np.zeros((1, self.system.num_states))  # Column 
        for experiment in range(experiments):
        # Generate excitation inputs (One psrb for each system input)
            for _n_input in range(self.system.num_inputs):
                _u_signal = self._get_psrb_signal(samples, sequence_limits=(5, 300), magnitude_limits=(-1, 1))
                u_vector[:, _n_input] = _u_signal.reshape(-1, 1)[:, 0]
            
            # Perform experiments
            x0 = np.random.uniform(0, 1, self.system.num_states)
            results = self.system.experiment(u_vector, x0, ts)

            # Obtained observed vector
            y_vector[:, state] = results[:, state]
            # plt.plot(y_vector[:10000])
            # plt.show()

            dummy_y = np.vstack((dummy_y, y_vector))
            dummy_u = np.vstack((dummy_u, u_vector))
        
        # Remove zero columns
        u_vector = np.delete(dummy_u, 0, axis=0)
        y_vector = np.delete(dummy_y, 0, axis=0)
        plt.plot(u_vector[:1000])
        plt.plot(y_vector[:1000])
        plt.show()
        
        # Normalize data
        u_mean = np.mean(u_vector)
        u_std = np.std(u_vector)
        y_mean = np.mean(y_vector)
        y_std = np.std(y_vector)

        u_vector = (u_vector - u_mean) / u_std
        y_vector = (y_vector - y_mean) / y_std
        self.scale_parameters = {"u_mean": u_mean, "u_std":u_std, "y_mean":y_mean, "y_std":y_std}

        # Autoregressive data
        u_vector, y_vector = self._construct_regression_vector(u_vector, y_vector, delay=delay)
        
        # Split data
        _train_size = int (len(u_vector) * 0.80)
        u_vector_train, u_vector_test = u_vector[:_train_size], u_vector[_train_size:]
        y_vector_train, y_vector_test = y_vector[:_train_size], y_vector[_train_size:]

        # Resize for deep learning model (samples, features, sequences)
        u_vector_train = u_vector_train.reshape((-1, u_vector_train.shape[1], 1))
        u_vector_test = u_vector_test.reshape((-1, u_vector_test.shape[1], 1))

        y_vector_train = y_vector_train.reshape((-1, y_vector_train.shape[1], 1))
        y_vector_test = y_vector_test.reshape((-1, y_vector_test.shape[1], 1))

        # Create datasets
        generator_train = tf.data.Dataset.from_tensor_slices((u_vector_train, y_vector_train) ).batch(batch_size)
        generator_test = tf.data.Dataset.from_tensor_slices((u_vector_test, y_vector_test)).batch(batch_size)
        print(generator_train.element_spec)

        # Create datasets
        # generator_train = TimeseriesGenerator(u_vector_train, y_vector_train, length=1, batch_size=batch_size, shuffle=False)
        # generator_test = TimeseriesGenerator(u_vector_test, y_vector_test, length=1, batch_size=batch_size, shuffle=False)
        # generator_train = generator_train.make_one_shot_iterator()
        # generator_test = generator_test.make_one_shot_iterator()
        

        return generator_train, generator_test

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
        # Remove u_vector from new_u (Not lagged version)
        new_u = np.delete(new_u, 0, axis=1)

        for _ in range(delay):
            # Roll Y vector 
            _aux_y = np.roll(_aux_y, 1, axis=0)
            # Fill with zeroes or Initial conditions TODO
            _aux_y[0] = 0
            new_u = np.hstack((new_u, _aux_y))
        return new_u, new_y
