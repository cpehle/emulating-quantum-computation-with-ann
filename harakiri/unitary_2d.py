import numpy as np

from . import quantum as qm
from . import transform as tfm
from . import random as rnd

from .layers.quantized import quantized_layers as qnt

import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential

def generate_data(unitary_transform, num_samples=1000):
  """
  Generate a set of training data for learning a given unitary
  transformation in U(2).

  Args:
    unitary_transform: Unitary transformation to be used.
    num_samples: Number of samples to be generated.
  """
  initial_angles = rnd.theta_phi_within(
      num_samples, 
      phi_low=0.0, 
      phi_high=0.1, 
      theta_low=np.pi/2, 
      theta_high=np.pi + 0.3
  )
  psi_initial = qm.psi(theta = initial_angles[:,0], phi = initial_angles[:,1])
  psi_final = np.matmul(unitary_transform, psi_initial)
  x = tfm.real_of_complex(psi_initial.T)
  y = tfm.real_of_complex(psi_final.T)
  return x, y

def build_linear_model(units, optimizer='rmsprop'):
  """
  Build a multi-layer linear regression model using 
  mean squared error as a loss function.

  Args:
    units: Dimensions of the units to be used.
    optimizer: Optimizer to be used.
  """
  model = Sequential()
  for unit in units:
    model.add(Dense(
      units=unit,
      use_bias=False,
      activation='linear'
    ))
  model.compile(
    loss="mse", 
    optimizer=optimizer,
    metrics=['accuracy']
  )
  return model

def build_non_linear_model(units, activation='relu', optimizer='rmsprop'):
  """
  Build a multi-layer non-linear regression model using 
  mean squared error as a loss function.

  Args:
    units: List of dimensions of the units to be used
    activation: activation function to be used
    optimizer: optimizer to be used
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

def build_non_linear_quantized_model(units, num_bits=4, activation='relu', optimizer='rmsprop'):
  """
  Build a multi-layer non-linear regression model using 
  mean squared error as a loss function.

  Args:
    units: List of dimensions of the units to be used
    num_bits: Number of bits the weights are supposed to be quantized to.
    activation: activation function to be used
    optimizer: optimizer to be used
  """
  model = Sequential()
  for unit in units:
    model.add(qnt.QuantizedDense(
      units=unit,
      nb=num_bits,
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
    num_samples=1000,
    batch_size=512,
    model=build_linear_model(units=[4]),
    plot_losses=True
  ):
  """
  Train a model for a given constant unitary transformation. Per default
  we are attempting to fit a linear model with one layer to the given
  data.

  Args:
    unitary_transform: Unitary transformation to be fit.
    epochs: Number of epochs to be trained.
    num_samples: Number of training samples to be generated.
    batch_size: Batch size to be used.
    model: Model to be trained.
  """
  x,y = generate_data(unitary_transform, num_samples=num_samples)

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
    plt.legend()
    plt.show()

  return model, history