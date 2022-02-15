import numpy as np
import tensorflow as tf


class ModelClass:
    def __init__(self, **kwargs) -> None:
        self._udelay = kwargs.pop("u_delay")
        self._num_states = kwargs.pop("num_states")
        self._num_features = kwargs["num_features"]
        self.model = self._generate_model(**kwargs)
        pass

    def _setup(self):
        return

    @staticmethod
    def _generate_model():
        return

    def train(self, data, steps_per_epoch=1, epochs=100, verbose=1):
        self.model.fit(data, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=verbose)
        return

    def get_summary(self):
        return self.model.summary()

    def predict(self, num_steps=1, x0=None, u=None):
        """
        Generates a future prediction for a given number of steps
        :param num_steps (int): Steps into the future t+n_steps
        :param x0 (ndarray):    Initial conditions, if not given are set to zero
        :param u (ndarray):     Input vector
        return prediction array
        """
        if x0 is None:
            x0 = 0 # TODO
        # Construct u
        u = np.zeros(self._udelay) 
        u[0] = 1.0
        # Construct y
        y = np.ones(self._num_states*self._udelay)
        for _ in range(self._num_states):
                y[_] = x0[_]
        # Construct to nn
        nn_input = np.concatenate((u, y))
        
        final_result = np.zeros((num_steps, self._num_states))
        for step in range(num_steps):
            y_hat = self.model(nn_input.reshape(-1, 1, self._num_features))
            # Update nn_input with feedback
            y = np.roll(y, self._num_states, axis=0)
            u = np.roll(u, 1, axis=0)
            u[0] = 1.0
            for _ in range(self._num_states):
                if self._num_states == 1:
                    y[_] = tf.squeeze(y_hat)
                else:
                    y[_] = tf.squeeze(y_hat)[_]
            nn_input = np.concatenate((u, y))
            final_result[step] =  tf.squeeze(y_hat)
        return final_result

