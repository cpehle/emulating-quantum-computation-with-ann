import numpy as np

from . import random as rnd
from . import quantum as qm
from . import transform as tfm

import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential

import matplotlib.pyplot as plt

def generate_data(u=qm.hadamard, n=2, num_samples=10000):
    """
    Args:
      u (quantum gate): quantum gate.
      n (int): Dimension of the quantum gate.
      m (int): Number of samples.
    """
    x = rnd.density_matrix_ginibre(n=n, m=num_samples)
    y = np.matmul(np.matmul(u, x), u.conj().T)
    # reshape and flatten data
    #eigenvalues, traces, eigenvectors = qm.eigen_decomposition(x)
    #print(np.mean(traces))
    x_real = np.reshape(np.stack([tfm.complex_matrix_to_real(m) for m in x]), (num_samples,(2*n)**2))
    y_real = np.reshape(np.stack([tfm.complex_matrix_to_real(m) for m in y]), (num_samples,(2*n)**2))
    return x_real, y_real

def build_non_linear_model(units, activation='linear', optimizer='adagrad'):
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

def train_model(
  unitary_transform = qm.hadamard, 
  name='',
  epochs=3000, 
  num_samples=10000,
  batch_size=1000,
  model=build_non_linear_model(units=[16,32,32,16]),
  plot_losses=True,
  num_subepochs=1000,
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
    x,y = generate_data(unitary_transform, n=np.shape(unitary_transform)[0], num_samples=num_samples)
  else:
    x,y = data
  # define call backs for tensorboard visualization etc.
  dim = np.shape(unitary_transform)[0]
  # batch_prediction_history = BatchPredictionHistory(x)

  callbacks = []
  if False:
    tensorboard_cb = keras.callbacks.TensorBoard(log_dir='logs/', histogram_freq=100)
    modelcheckpoint_cb = keras.callbacks.ModelCheckpoint(filepath='checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', period=100)
    callbacks = [tensorboard_cb, modelcheckpoint_cb]
  
  mean_traces = []
  variance_traces = []
  validation_losses = []
  training_losses = []
  sub_epochs = int(epochs/num_subepochs)

  for _ in range(num_subepochs):
    y_pred = model.predict(x=x)
    y_pred = [np.reshape(ys, (2*dim, 2*dim)) for ys in y_pred]
    y_pred = [tfm.complex_matrix_to_real(ys) for ys in y_pred]
    eigenvalues, traces, eigenvectors = qm.eigen_decomposition(y_pred)
    mean_traces.append(np.mean(traces))
    variance_traces.append(np.var(traces))
  
    
    history = model.fit(
      x=x,
      y=y,
      batch_size=batch_size,
      epochs=sub_epochs,
      validation_split=0.05,
      callbacks=callbacks
    )

    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    training_losses.append(training_loss)
    validation_losses.append(validation_loss)

  print(mean_traces)
  print(variance_traces)

  if plot_losses:
    plt.figure()
    plt.plot(training_losses, label='training loss')
    plt.yscale('log')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title('{}'.format(name))
    plt.legend()
    plt.savefig('figures/{}.png'.format(name))
    plt.show()

  for layer in model.layers:
    print(layer.get_config())
    print(layer.get_weights())

  return model, training_losses, validation_losses


def generate_loss_sweep(
    gate = qm.hadamard,
    units = [16,3,16],
    num_runs=100,
    num_samples = 10000,
    batch_size = 1000,
    epochs = 300,
    num_subepochs = 100,
    activation = 'linear'
  ):
  training_losses = []
  for _ in range(num_runs):
    _, training_loss, validation_loss = train_model(
        unitary_transform=gate, 
        name='density matrix - hadamard gate',
        epochs=epochs,
        model=build_non_linear_model(units=units, activation=activation),
        plot_losses=False,
        num_samples=num_samples,
        num_subepochs=num_subepochs,
        batch_size=batch_size
      )
    training_losses.append(training_loss)
  return training_losses

def generate_loss_sweep_plot(
    name='density matrix - hadamard gate', 
    title='', 
    gate = qm.hadamard,
    units = [16,3,16], 
    num_runs = 2,
    num_samples = 10000,
    epochs = 300,
    batch_size = 1000,
    num_subepochs = 100,
  ):
  training_losses = generate_loss_sweep(gate, units=units, batch_size=batch_size, num_runs=num_runs, epochs=epochs, num_subepochs=num_subepochs)
  plt.figure()
  plt.title(title)
  plt.yscale('log')
  plt.xlabel('steps')
  plt.ylabel('loss')
  for loss in training_losses:
    plt.plot(loss, alpha=0.3, color='b')
  plt.savefig('sweep_{}.png'.format(name))