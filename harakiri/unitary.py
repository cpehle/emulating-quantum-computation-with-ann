from . import random as rnd
from . import transform as tfm
from . import quantum as qm
from .layers.quantized import quantized_layers as qnt

import numpy as np
import pandas as pd

import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential

def generate_data(unitary_transform, num_samples=1000):
  """
  Generate training samples for the given unitary transformation.
 
  Args:
    num_samples (int): Number of samples to generate.
  """
  dim = np.shape(unitary_transform)[0]
  assert(dim == 4)

  phi,theta = rnd.uniform_2d_spherical(m = num_samples)
  psi_0_initial = qm.psi(theta = theta, phi = phi)
  phi,theta = rnd.uniform_2d_spherical(m = num_samples)
  psi_1_initial = qm.psi(theta = theta, phi = phi)
  psi_initial = np.vstack((psi_0_initial, psi_1_initial))

  psi_final = np.matmul(unitary_transform, psi_initial)

  x_0 = tfm.real_of_complex(psi_0_initial.T)
  x_1 = tfm.real_of_complex(psi_1_initial.T)
  x = np.hstack((x_0, x_1))

  psi_0_final = psi_final[:2,:]
  psi_1_final = psi_final[2:,:]

  y_0 = tfm.real_of_complex(psi_0_final.T)
  y_1 = tfm.real_of_complex(psi_1_final.T)

  y = np.hstack((y_0, y_1))
  return x, y

def build_non_linear_model(units, activation='relu', optimizer='adam'):
  """
  Build a multi-layer non-linear regression model using 
  mean squared error as a loss function.

  Args:
    units: List of dimensions of the units to be used.
    activation: Activation function to be used.
    optimizer: Optimizer to be used.
  """
  model = Sequential()
  for unit in units:
    model.add(Dense(
      units=unit,
      use_bias=True,
      activation=activation
    ))
  model.compile(
    loss="mse", 
    optimizer=optimizer,
    metrics=['accuracy']
  )
  return model

def build_non_linear_quantized_model(units, activation='relu', optimizer='adam'):
  """
  Build a multi-layer non-linear quantized regression model using 
  mean squared error as a loss function.

  Args:
    units: List of dimensions of the units to be used.
    activation: Activation function to be used.
    optimizer: Optimizer to be used.
  """
  model = Sequential()
  for unit in units:
    model.add(qnt.QuantizedDense(
      units=unit,
      use_bias=True,
      activation=activation
    ))
  model.compile(
    loss="mse", 
    optimizer=optimizer,
    metrics=['accuracy']
  )
  return model

def train_model(
    unitary_transform, 
    epochs=3000, 
    num_samples=5000,
    batch_size=5000,
    model=build_non_linear_model(units=[16,16,8], activation='linear'),
    plot_losses=True,
    data=None
  ):
  """
  Train a model for a given constant unitary transformation. Per default
  we are attempting to fit a linear model with one layer to the given
  data.

  Note: This is a copy of train in unitary_2d, should merge.

  Args:
    unitary_transform: Unitary transformation to be fit.
    epochs: Number of epochs to be trained.
    num_samples: Number of training samples to be generated.
    batch_size: Batch size to be used.
    model: Model to be trained.
  """
  if data is None:
    x,y = generate_data(unitary_transform, num_samples=num_samples)
  else:
    x,y = data

  # define call backs for tensorboard visualization etc.
  callbacks = []
  if False:
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir='logs/', histogram_freq=100)
    modelcheckpoint_cb = keras.callbacks.ModelCheckpoint(filepath='checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=100)
    callbacks = [tensorboard_cb, modelcheckpoint_cb]
  
  history = model.fit(
    x=x,
    y=y,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.05,
    callbacks=callbacks
  )

  if plot_losses:
    import matplotlib.pyplot as plt
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    plt.plot(training_loss, label='training loss')
    plt.plot(validation_loss, label='validation loss')
    plt.yscale('log')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

  return model, history

def generate_sweep_plots(error_style='unit_traces', num_runs=100, quantize=False):
  training_losses = []
  epochs = 6000
  for i in range(num_runs):
    if quantize:
      model = build_non_linear_quantized_model(units=[16,16,8], activation='linear')
    else:
      model = build_non_linear_model(units=[16,16,8], activation='linear')
    model, history = train_model(
      unitary_transform=qm.cnot, 
      epochs=epochs, 
      plot_losses=False,
      model=model
    )
    training_losses.append(history.history['loss'])

  import matplotlib.pyplot as plt
  import seaborn as sns
  plt.figure()
  plt.title('CNOT Gate Linear 4Bit-Quantized 16-16-8')
  plt.yscale('log')
  plt.xlabel('steps')
  plt.ylabel('loss')
  sns.tsplot(np.array(training_losses), err_style=error_style)
  plt.savefig('sweep_{}.png'.format('cnot'))

if __name__ == '__main__':
  pass
  





