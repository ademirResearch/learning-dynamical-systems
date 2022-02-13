from model_class import ModelClass
from keras.models import Sequential, Model
from keras.layers import LSTM, TimeDistributed, Input, Dense

class AutoEncoder(ModelClass):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @staticmethod
    def _generate_model(num_encoder_tokens=1, num_decoder_tokens=1, latent_dim=256):
        # MODEL FOR ENCODER AND DECODER -------------------------------------------
        # encoder training
        encoder_inputs = Input(shape = (None, num_encoder_tokens))
        encoder = LSTM(latent_dim, 
                batch_input_shape = (1, None, num_encoder_tokens),
                stateful = False,
                return_sequences = True,
                return_state = True,
                recurrent_initializer = 'glorot_uniform')

        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c] 

        # Decoder training, using 'encoder_states' as initial state.
        decoder_inputs = Input(shape=(None, num_encoder_tokens))

        decoder_lstm_1 = LSTM(latent_dim,

                batch_input_shape = (1, None, num_encoder_tokens),
                stateful = False,
                return_sequences = True,
                return_state = False,
                dropout = 0.4,
                recurrent_dropout = 0.4) # True

        decoder_lstm_2 = LSTM(128, 
                    stateful = False,
                    return_sequences = True,
                    return_state = True,
                    dropout = 0.4,
                    recurrent_dropout = 0.4)

        decoder_outputs, _, _ = decoder_lstm_2(decoder_lstm_1(decoder_inputs, initial_state = encoder_states))
        decoder_dense = TimeDistributed(Dense(num_decoder_tokens, activation = 'relu'))
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # Train the model
        training_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

        return training_model