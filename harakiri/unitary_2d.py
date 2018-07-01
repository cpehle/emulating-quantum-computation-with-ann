import numpy as np

from . import quantum as qm
from . import transform as tfm
from . import random as rnd

from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential

def generate_data(unitary_transform, num_samples=1000):
  """
  Generate a set of training data for learning a given unitary
  transformation in U(2).
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
  """
  model = Sequential()
  for unit in units:
    model.add(Dense(
      units=unit,
      use_bias=False,
      activation='linear'
    ))
  model.compile(loss="mse", optimizer=optimizer)
  return model

def build_non_linear_model(units, activation, optimizer='rmsprop'):
  """
  Build a multi-layer non-linear regression model using 
  mean squared error as a loss function.
  """
  model = Sequential()
  for unit in units:
    model.add(Dense(
      units=unit,
      use_bias=True,
      activation=activation
    ))
  model.compile(loss="mse", optimizer=optimizer)
  return model

def train_model(
    unitary_transform, 
    epochs=3000, 
    num_samples=1000,
    model=build_linear_model(units=[4])
  ):
  """
  Train a model for a given constant unitary transformation. Per default
  we are attempting to fit a linear model with one layer to the given
  data.
  """
  x,y = generate_data(unitary_transform, num_samples=num_samples)
  model.fit(
    x=x,
    y=y,
    batch_size=num_samples,
    epochs=epochs,
    validation_split=0.05
  )
  return model

