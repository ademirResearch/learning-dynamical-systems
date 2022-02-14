import numpy as np
import tensorflow as tf


class ModelClass:
    def __init__(self, **kwargs) -> None:
        self._udelay = kwargs.pop("u_delay")
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
        u = np.zeros(self._udelay + 1) 
        u[0] = 1.0
        # Construct y
        y = np.ones(3 + 2*self._udelay - 1)
        for _ in range(3):
                y[_] = x0[_]
        # Construct to nn
        nn_input = np.concatenate((u, y))
        # Reshape
        # nn_input = nn_input.reshape(-1, self._udelay, 1)

        final_result = np.zeros((num_steps, 3))
        for step in range(num_steps):
            y_hat = self.model(nn_input.reshape(-1, 1, 9))
            # Update nn_input with feedback
            y = np.roll(y, 3, axis=0)
            u = np.roll(u, 1, axis=0)
            u[0] = 1.0
            for _ in range(3):
                y[_] = tf.squeeze(y_hat)[_]
            nn_input = np.concatenate((u, y))
            final_result[step] =  tf.squeeze(y_hat)
        return final_result

