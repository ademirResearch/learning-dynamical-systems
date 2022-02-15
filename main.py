#%%
from systems import Lorenz
import matplotlib.pyplot as plt
from data_generator import DynamicSystemDataGenerator
from autoencoder import AutoEncoder
from mlp import MLP

sys = Lorenz()
result = sys.step(num_steps=200, ts=0.01, x0=[1.0, 1.0, 1.0])
# plt.plot(result)
# plt.show()

delay = 20
num_states = 1
num_inputs = 1
data_manager = DynamicSystemDataGenerator(system=Lorenz())
_data = data_manager.get_data_generator(samples=5000, ts=0.01, delay=delay, batch_size=1, state="1")

# params = {"num_encoder_tokens": 1, "num_decoder_tokens": 1, "latent_dim": 256}
params = {"num_inputs": num_inputs, 
          "num_features": (num_inputs*delay + num_states*delay), 
          "u_delay": delay, 
          "num_states": num_states}
# autoencoder = AutoEncoder(**params)
mlp = MLP(**params)
print(mlp.get_summary())
mlp.train(_data, steps_per_epoch=1, epochs=100, verbose=1)
prediction = mlp.predict(num_steps=200, u=1.0, x0=[1.0, 1.0, 1.0])
plt.plot(prediction)
plt.grid()
plt.show()
# %%
