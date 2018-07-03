from . import random as rnd
from . import transform as tfm

import numpy as np

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
  psi_initial = rnd.complex_projective_spherical(n = dim, m = num_samples)
  psi_final = np.matmul(unitary_transform, psi_initial)
  x = tfm.real_of_complex(psi_initial.T)
  y = tfm.real_of_complex(psi_final.T)
  return x, y

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

def train_model(
    unitary_transform, 
    epochs=3000, 
    num_samples=1000,
    batch_size=512,
    model=build_non_linear_model(units=[4]),
    plot_losses=True
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
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

  return model, history





