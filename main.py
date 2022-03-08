#%%
from systems import Lorenz, FirstOrder
import matplotlib.pyplot as plt
from data_generator import DynamicSystemDataGenerator
from autoencoder import AutoEncoder
from mlp import MLP
from lstm import RNN
from sklearn.metrics import mean_squared_error

# sys = FirstOrder()
# result = sys.step(num_steps=1000, ts=0.01, x0=[0.0])
# plt.plot(result)
# plt.show()

delay = 2
num_states = 1
num_inputs = 1
data_manager = DynamicSystemDataGenerator(system=FirstOrder())
_train, _test = data_manager.get_data_generator(samples=100000, experiments=1, ts=0.02, delay=delay, batch_size=1, state="0")

# params = {"num_encoder_tokens": 1, "num_decoder_tokens": 1, "latent_dim": 256}
params = {"num_inputs": num_inputs, 
          "num_features": (num_inputs*delay + num_states*delay), 
          "u_delay": delay, 
          "num_states": num_states}
# autoencoder = AutoEncoder(**params)
mlp = RNN(**params)
print(mlp.get_summary())
mlp.train(_train, _test, steps_per_epoch=1, epochs=1000, verbose=1)

plt.plot(mlp.history.history["loss"])
plt.plot(mlp.history.history["val_loss"])
plt.legend(["loss", "val"])
plt.show()


# Testing
u = 1 - data_manager.scale_parameters["u_mean"] / data_manager.scale_parameters["u_std"]
ic = 0 - data_manager.scale_parameters["y_mean"] / data_manager.scale_parameters["y_std"]


sys = FirstOrder()
result = sys.step(num_steps=200, ts=0.02, x0=[0.0])
plt.plot(result)
prediction = mlp.predict(num_steps=200, u=u, x0=[ic])
plt.plot(prediction)
plt.legend(["ground_truth", "prediction"])
plt.grid()
plt.show()

print(mean_squared_error(result, prediction[:200], squared=False))

# %%

