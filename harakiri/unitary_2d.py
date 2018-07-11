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
  #initial_angles = rnd.theta_phi_within(
  #    num_samples, 
  #    phi_low=0.0, 
  #    phi_high=0.1, 
  #    theta_low=np.pi/2, 
  #    theta_high=np.pi + 0.3
  #)
  #psi_initial = qm.psi(theta = initial_angles[:,0], phi = initial_angles[:,1])
  phi,theta = rnd.uniform_2d_spherical(m = num_samples)
  psi_initial = qm.psi(theta = theta, phi = phi)
  psi_final = np.matmul(unitary_transform, psi_initial)
  x = tfm.real_of_complex(psi_initial.T)
  y = tfm.real_of_complex(psi_final.T)
  return x, y


def generate_shifted_data(unitary_transform, num_samples=1000, eps = 0.01):
  """
  Generate shifted training samples for the given unitary transformation. 

  Args:
    unitary_transform: Unitary transformation to generate samples for.
    num_samples (int): Number of samples to generate.
    eps (float): Additional offset from 0.
  """
  x,y = generate_data(unitary_transform=unitary_transform, num_samples=num_samples)
  shift_x = np.amax(np.abs(x)) * np.ones_like(x) + eps
  shift_y = np.amax(np.abs(y)) * np.ones_like(y) + eps
  return x + shift_x, y + shift_y


def build_linear_model(units, optimizer='adam'):
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

def build_non_linear_model(units, activation='relu', optimizer='adam'):
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

def build_non_linear_quantized_model(units, num_bits=4, activation='relu', optimizer='adam'):
  """
  Build a multi-layer non-linear quantized regression model using 
  mean squared error as a loss function.

  Args:
    units: List of dimensions of the units to be used.
    num_bits: Number of bits the weights are supposed to be quantized to.
    activation: Activation function to be used.
    optimizer: Optimizer to be used.
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
    name='',
    epochs=3000, 
    num_samples=1000,
    batch_size=512,
    model=build_linear_model(units=[4]),
    plot_losses=True,
    data=None
  ):
  """
  Train a model for a given constant unitary transformation. Per default
  we are attempting to fit a linear model with one layer to the given
  data.

  Args:
    unitary_transform: Unitary transformation to be fit.
    name: Name of the model.
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
    plt.figure()
    plt.plot(training_loss, label='training loss')
    plt.plot(validation_loss, label='validation loss')
    plt.yscale('log')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title('{}'.format(name))
    plt.legend()
    plt.savefig('figures/{}.png'.format(name))

  return model, history

def generate_loss_sweep(
    gate = qm.hadamard, 
    units = [8,8,4], 
    num_runs=100, 
    quantize=False,
    num_samples=1000,
    batch_size=512
  ):
  epochs = 1500
  training_losses = []
  for i in range(num_runs):
    if not quantize:
      model, history = train_model(
          unitary_transform=gate, 
          name='',
          epochs=epochs,
          model=build_non_linear_model(units=units, activation='linear'),
          plot_losses=False,
          num_samples=num_samples,
          batch_size=batch_size
        )
      training_losses.append(history.history['loss'])
    else:
      model, history = train_model(
          unitary_transform=gate, 
          name='',
          epochs=epochs,
          model=build_non_linear_quantized_model(units=units, activation='linear'),
          plot_losses=False,
          num_samples=num_samples,
          batch_size=batch_size
        )
      training_losses.append(history.history['loss'])
  return training_losses

def generate_loss_sweep_plot(
    name='hadamard', 
    title='', 
    gate = qm.hadamard,
     units = [8,8,4], 
     num_runs=100, 
     quantize = False,
     num_samples = 1000
  ):
  training_losses = generate_loss_sweep(gate, units, num_runs, quantize=quantize)
  import matplotlib.pyplot as plt
  import seaborn as sns
  plt.figure()
  plt.title(title)
  plt.yscale('log')
  plt.xlabel('steps')
  plt.ylabel('loss')
  sns.tsplot(np.array(training_losses), err_style='unit_traces')
  plt.savefig('sweep_{}.png'.format(name))

def generate_all_plots(num_runs = 5):
  # generate_loss_sweep_plot(
  #   name='hadamard', 
  #   title='Hadamard Gate Linear 8-8-4', 
  #   gate=qm.hadamard, 
  #   units=[8,8,4],
  #   num_runs=num_runs
  # )
  # generate_loss_sweep_plot(
  #   name='rotate_half', 
  #   title='Rotate($\pi/2$) Linear 8-8-4', 
  #   gate=qm.rotate(np.pi/2), 
  #   units=[8,8,4],
  #   num_runs=num_runs
  # )
  # generate_loss_sweep_plot(
  #   name='rotate_quarter', 
  #   title='Rotate($\pi/4$) Linear 8-8-4', 
  #   gate=qm.rotate(np.pi/4), 
  #   units=[8,8,4],
  #   num_runs=num_runs
  # )
  # generate_loss_sweep_plot(
  #   name='rotate_eigth', 
  #   title='Rotate($\pi/8$) Linear 8-8-4', 
  #   gate=qm.rotate(np.pi/8), 
  #   units=[8,8,4],
  #   num_runs=num_runs
  # )
  # generate_loss_sweep_plot(
  #   name='rotate_sixteenth', 
  #   title='Rotate($\pi/16$) Linear 8-8-4', 
  #   gate=qm.rotate(np.pi/16), 
  #   units=[8,8,4],
  #   num_runs=num_runs
  # )
  # generate_loss_sweep_plot(
  #   name='x_gate', 
  #   title='Pauli-X Gate Linear 8-8-4', 
  #   gate=qm.sigma_1, 
  #   units=[8,8,4],
  #   num_runs=num_runs
  # )
  # generate_loss_sweep_plot(
  #   name='z_gate', 
  #   title='Pauli-Z Gate Linear 8-8-4', 
  #   gate=qm.sigma_3, 
  #   units=[8,8,4],
  #   num_runs=num_runs
  # )

  # quantized plots

  # generate_loss_sweep_plot(
  #   name='hadamard-quantized', 
  #   title='Hadamard Gate 4Bit-Quantized Linear 8-8-4', 
  #   gate=qm.hadamard, 
  #   units=[8,8,4],
  #   num_runs=num_runs,
  #   quantize=True
  # )
  # generate_loss_sweep_plot(
  #   name='rotate_half-quantized', 
  #   title='Rotate($\pi/2$) 4Bit-Quantized Linear 8-8-4', 
  #   gate=qm.rotate(np.pi/2), 
  #   units=[8,8,4],
  #   num_runs=num_runs,
  #   quantize=True
  # )
  # generate_loss_sweep_plot(
  #   name='rotate_quarter-quantized', 
  #   title='Rotate($\pi/4$) 4Bit-Quantized Linear 8-8-4', 
  #   gate=qm.rotate(np.pi/4), 
  #   units=[8,8,4],
  #   num_runs=num_runs,
  #   quantize=True
  # )
  # generate_loss_sweep_plot(
  #   name='rotate_eigth-quantized', 
  #   title='Rotate($\pi/8$) 4Bit-Quantized Linear 8-8-4', 
  #   gate=qm.rotate(np.pi/8), 
  #   units=[8,8,4],
  #   num_runs=num_runs,
  #   quantize=True
  # )
  # generate_loss_sweep_plot(
  #   name='rotate_sixteenth-quantized', 
  #   title='Rotate($\pi/16$) 4Bit-Quantized Linear 8-8-4', 
  #   gate=qm.rotate(np.pi/16), 
  #   units=[8,8,4],
  #   num_runs=num_runs,
  #   quantize=True
  # )
  # generate_loss_sweep_plot(
  #   name='x_gate-quantized', 
  #   title='Pauli-X Gate 4Bit-Quantized Linear 8-8-4', 
  #   gate=qm.sigma_1, 
  #   units=[8,8,4],
  #   num_runs=num_runs,
  #   quantize=True
  # )
  # generate_loss_sweep_plot(
  #   name='z_gate-quantized', 
  #   title='Pauli-Z Gate 4Bit-Quantized Linear 8-8-4', 
  #   gate=qm.sigma_3, 
  #   units=[8,8,4],
  #   num_runs=num_runs,
  #   quantize=True
  # )
  generate_loss_sweep_plot(
    name='hadamard-quantized', 
    title='Hadamard Gate 4Bit-Quantized Linear 16-16-4', 
    gate=qm.hadamard, 
    units=[16,16,4],
    num_runs=num_runs,
    quantize=True,
    num_samples=10000,
    batch_size=5000
  )