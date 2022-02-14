from model_class import ModelClass
from keras.models import Sequential, Model
from keras.layers import Input, Dense

class MLP(ModelClass):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def _generate_model(num_inputs=1, num_features=1):
        model = Sequential()
        model.add(Dense(1024, activation='tanh', input_shape=(num_inputs, num_features)))
        model.add(Dense(3))
        model.compile(optimizer='adam', loss='mse')
        return model