#%%
from systems import Lorenz
import matplotlib.pyplot as plt
from data_generator import DynamicSystemDataGenerator
from autoencoder import AutoEncoder
from mlp import MLP

# sys = Lorenz()
# result = sys.step(num_steps=200, ts=0.01, x0=[1.0, 1.0, 1.0])
# plt.plot(result)
# plt.show()

data_manager = DynamicSystemDataGenerator(system=Lorenz())
_data = data_manager.get_data_generator(samples=100, ts=0.01, delay=2, batch_size=1)

# params = {"num_encoder_tokens": 1, "num_decoder_tokens": 1, "latent_dim": 256}
params = {"num_inputs": 2, "num_features": 1}
# autoencoder = AutoEncoder(**params)
mlp = MLP(**params)
print(mlp.get_summary())
mlp.model.fit(_data, steps_per_epoch=1, epochs=100, verbose=1)

# %%
