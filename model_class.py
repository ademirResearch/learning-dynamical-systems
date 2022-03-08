from random import shuffle
import numpy as np
import tensorflow as tf


class ModelClass:
    def __init__(self, **kwargs) -> None:
        self._udelay = kwargs.pop("u_delay")
        self._num_states = kwargs.pop("num_states")
        self._num_features = kwargs["num_features"]
        self.history = None
        self.model = self._model(**kwargs)
        self._compile()
        pass

    def _setup(self):
        return

    def _model(self):
        return

    def _compile(self):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=["mse"])
        return

    def train(self, data, validation, steps_per_epoch=1, epochs=100, verbose=1):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
        self.history = self.model.fit(data, validation_data=validation, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=verbose, callbacks=[callback], shuffle=True)
        return

    def get_summary(self):
        return self.model.summary()

    def predict(self, x0, num_steps=1, u=None):
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
        u0 = u
        u = np.zeros(self._udelay) 
        u[0] = u0
        # Construct y
        y = np.zeros(self._num_states*self._udelay)
        for _ in range(self._num_states):
                y[_] = x0[_]
        # Construct to nn
        nn_input = np.concatenate((u, y))
        
        final_result = np.zeros((num_steps + 1, self._num_states))
        for step in range(num_steps):
            y_hat = self.model.predict(nn_input.reshape(-1, self._num_features, 1))
            # Update nn_input with feedback
            y = np.roll(y, self._num_states, axis=0)
            u = np.roll(u, 1, axis=0)
            u[0] = u0
            for _ in range(self._num_states):
                if self._num_states == 1:
                    y[_] = tf.squeeze(y_hat)
                else:
                    y[_] = tf.squeeze(y_hat)[_]
            nn_input = np.concatenate((u, y))
            final_result[step + 1] =  tf.squeeze(y_hat)
        return final_result

