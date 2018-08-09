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
    metrics=[]
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
  data=None,
  verbose=1,
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
    model: Model to be trained
    plot_losses (bool): Create a plot of the models loss.
    num_subepochs (int): Number of training steps to divide it into.
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
  std_traces = []
  std_hermitian_parts = []
  mean_hermitian_parts = []
  std_anti_hermitian_parts = []
  mean_anti_hermitian_parts = []
  validation_losses = []
  training_losses = []
  sub_epochs = int(epochs/num_subepochs)

  for _ in range(num_subepochs):
    y_pred = model.predict(x=x)
    y_pred = [np.reshape(ys, (2*dim, 2*dim)) for ys in y_pred]
    y_pred = [tfm.real_matrix_to_complex(ys) for ys in y_pred]
    eigenvalues, traces, eigenvectors = qm.eigen_decomposition(y_pred)
    hermitian_part = [qm.hermitian_part(ys) for ys in y_pred]
    hermitian_part_norm = [np.real(np.trace(np.matmul(ys.conj().T, ys))) for ys in hermitian_part]
    anti_hermitian_part = [qm.anti_hermitian_part(ys) for ys in y_pred]
    anti_hermitian_part_norm = [np.real(np.trace(np.matmul(ys.conj().T, ys))) for ys in anti_hermitian_part]
    mean_traces.append(np.mean(traces))
    std_traces.append(np.std(traces))
    mean_hermitian_parts.append(np.mean(hermitian_part_norm))
    std_hermitian_parts.append(np.std(hermitian_part_norm))
    mean_anti_hermitian_parts.append(np.mean(anti_hermitian_part_norm))
    std_anti_hermitian_parts.append(np.std(anti_hermitian_part_norm))

    history = model.fit(
      x=x,
      y=y,
      batch_size=batch_size,
      epochs=sub_epochs,
      validation_split=0.05,
      callbacks=callbacks,
      verbose=verbose
    )

    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    training_losses += training_loss
    validation_losses += validation_loss

  print(mean_traces)
  print(std_traces)

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

  return (model, training_losses, validation_losses, mean_traces, std_traces, mean_hermitian_parts, std_hermitian_parts, mean_anti_hermitian_parts, std_anti_hermitian_parts)

def compose_circuit(num_layers = 2):
  epochs = 300
  num_subepochs = 1
  num_samples = 10000
  units = [64,16,64]
  batch_size = 1000

  hadamard_pi_8th, h_loss, _, _, _, _, _, _, _ = train_model(
    unitary_transform=np.kron(qm.hadamard, qm.rotate(np.pi/8)),
    name='hadamard_rotate_pi_8th',
    epochs=epochs,
    model=build_non_linear_model(units=units, activation='linear'),
    plot_losses=False,
    num_samples=num_samples,
    num_subepochs=num_subepochs,
    batch_size=batch_size
  )
  cnot, cnot_loss, _, _, _, _, _, _, _ = train_model(
    unitary_transform=qm.cnot,
    name='cnot',
    epochs=epochs,
    model=build_non_linear_model(units=units, activation='linear'),
    plot_losses=False,
    num_samples=num_samples,
    num_subepochs=num_subepochs,
    batch_size=batch_size
  )
  
  unitary_transform = np.matmul(qm.cnot, np.kron(qm.hadamard, qm.rotate(np.pi/8)))
  unitary_transform = np.linalg.matrix_power(unitary_transform, num_layers)

  x,y = generate_data(unitary_transform, n=np.shape(unitary_transform)[0], num_samples=10)

  x_cnot = x
  for _ in range(num_layers):
    x_h_r = hadamard_pi_8th.predict(x=x_cnot)
    x_cnot = cnot.predict(x=x_h_r)

  print(x_cnot)
  print(y)
  


def generate_bottleneck_sweep(name = 'hadamard', gate = qm.hadamard):
  num_samples = 10000
  batch_size = 1000
  epochs = 1000
  num_subepochs = 1
  activation = 'linear'
  bottleneck_dim = np.arange(1,9)

  losses = []

  for dim in bottleneck_dim:
    _, training_loss, validation_loss, mean_trace, std_trace, mean_hermitian_part, std_hermitian_part, mean_anti_hermitian_part, std_anti_hermitian_part = train_model(
          unitary_transform=gate, 
          epochs=epochs,
          model=build_non_linear_model(units=[16,dim,16], activation=activation),
          plot_losses=False,
          num_samples=num_samples,
          num_subepochs=num_subepochs,
          batch_size=batch_size,
          verbose=0,
    )
    min_loss = np.min(training_loss)
    losses.append(min_loss)

  fig, axis = plt.subplots(1, 1)
  axis.set_title('Loss versus bottleneck dimension')
  axis.set_yscale('log')
  axis.set_xlabel('bottleneck dimension')
  axis.set_ylabel('loss')
  axis.plot(bottleneck_dim, losses)
  fig.tight_layout()
  fig.subplots_adjust(top=0.88)
  fig.savefig('bottle_neck_sweep_{}.png'.format(name))

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
  mean_traces = []
  std_traces = []
  mean_anti_hermitian_parts = []
  mean_hermitian_parts = []
  std_hermitian_parts = []
  std_anti_hermitian_parts = []

  for _ in range(num_runs):
    _, training_loss, validation_loss, mean_trace, std_trace, mean_hermitian_part, std_hermitian_part, mean_anti_hermitian_part, std_anti_hermitian_part = train_model(
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
    mean_traces.append(mean_trace)
    std_traces.append(std_trace)
    mean_anti_hermitian_parts.append(mean_anti_hermitian_part)
    std_anti_hermitian_parts.append(std_anti_hermitian_part)
    mean_hermitian_parts.append(mean_hermitian_part)
    std_hermitian_parts.append(std_hermitian_part)


  return training_losses, mean_traces, std_traces, mean_hermitian_parts, std_hermitian_parts, mean_anti_hermitian_parts, std_hermitian_parts

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
  training_losses, mean_traces, std_traces, mean_hermitian_parts, std_hermitian_parts, mean_anti_hermitian_parts, std_anti_hermitian_parts = generate_loss_sweep(gate, units=units, batch_size=batch_size, num_runs=num_runs, epochs=epochs, num_subepochs=num_subepochs)

  fig, axis = plt.subplots(3, 1)
  axis[0].set_title('Training progress')
  axis[0].set_yscale('log')
  axis[0].set_xlabel('steps')
  axis[0].set_ylabel('loss')
  for loss in training_losses:
    axis[0].plot(loss, alpha=0.3, color='b')

  axis[1].set_title('Average trace')
  axis[1].set_xlabel('steps')
  axis[1].set_ylabel('trace')
  for idx, mean_trace in enumerate(mean_traces):
    axis[1].errorbar(x=int(epochs/num_subepochs)*np.arange(0,num_subepochs), y=mean_trace, yerr=std_traces[idx], alpha=0.1, color='b')
  
  axis[2].set_title('Average Norm Hermitian / Anti-Hermitian part')
  axis[2].set_xlabel('steps')
  axis[2].set_ylabel('norm')
  #axis[2].set_yscale('log')
  for idx, mean_hermitian_part in enumerate(mean_hermitian_parts):
    axis[2].errorbar(x=int(epochs/num_subepochs)*np.arange(0,num_subepochs), y=mean_hermitian_part, yerr=std_hermitian_parts[idx], alpha=0.1, color='b')
  for idx, mean_anti_hermitian_part in enumerate(mean_anti_hermitian_parts):
    axis[2].errorbar(x=int(epochs/num_subepochs)*np.arange(0,num_subepochs), y=mean_anti_hermitian_part, yerr=std_anti_hermitian_parts[idx], alpha=0.1, color='r')

  fig.tight_layout()
  fig.subplots_adjust(top=0.88)
  fig.savefig('sweep_{}.png'.format(name))

def generate_plots():
  print("Hadamard")
  generate_loss_sweep_plot(title='', gate=qm.hadamard, units= [16,3,16], epochs=3000, num_subepochs=100)
  print("CNOT")
  generate_loss_sweep_plot(title='', gate=qm.cnot, units=[64,16,64], epochs=3000, num_subepochs=100)
