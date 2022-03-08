from model_class import ModelClass
from keras.models import Sequential, Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, LSTM

class RNN(ModelClass):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _model(self, num_inputs=1, num_features=1):
        model = Sequential()
        model.add(Input(shape=(num_features, 1)))
        model.add(LSTM(1024, activation="relu"))
        # model.add(Dropout(0.1))
        model.add(Dense(32, activation='tanh'))
        # model.add(Dropout(0.2))
        model.add(Dense(1))
        return model