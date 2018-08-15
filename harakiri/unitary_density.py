import numpy as np
import pickle

from . import random as rnd
from . import quantum as qm
from . import transform as tfm
from . import figures

import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential

import matplotlib.pyplot as plt
import matplotlib

normal_figure_width, wide_figure_width = figures.set_plot_parameters()

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

class TrainingResult:
  def __init__(self):
    self.training_losses = []
    self.validation_losses = []
    self.mean_traces = []
    self.std_traces = []
    self.mean_anti_hermitian_parts = []
    self.mean_hermitian_parts = []
    self.std_hermitian_parts = []
    self.std_anti_hermitian_parts = []

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
  dim = np.shape(unitary_transform)[0]

  callbacks = []
  result = TrainingResult()
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
    result.mean_traces.append(np.mean(traces))
    result.std_traces.append(np.std(traces))
    result.mean_hermitian_parts.append(np.mean(hermitian_part_norm))
    result.std_hermitian_parts.append(np.std(hermitian_part_norm))
    result.mean_anti_hermitian_parts.append(np.mean(anti_hermitian_part_norm))
    result.std_anti_hermitian_parts.append(np.std(anti_hermitian_part_norm))

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
    result.training_losses += training_loss
    result.validation_losses += validation_loss

  if plot_losses:
    plt.figure()
    plt.plot(result.training_losses, label='training loss')
    plt.yscale('log')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.title('{}'.format(name))
    plt.legend()
    plt.savefig('figures/{}.png'.format(name))
    plt.show()

  return model, result

def quantumness():
  epochs = 3000
  num_subepochs = 1
  num_samples = 10000
  units = [64,15,64]
  batch_size = 1000
  cnot, _ = train_model(
    unitary_transform=qm.cnot,
    name='cnot',
    epochs=epochs,
    model=build_non_linear_model(units=units, activation='linear'),
    plot_losses=False,
    num_samples=num_samples,
    num_subepochs=num_subepochs,
    batch_size=batch_size
  )
  num_test_samples = 10000
  dim = 4
  s = np.stack([rnd.ginibre_ensemble_sample(n = 4) for _ in range(num_test_samples)])
  s_real = np.reshape(np.stack([tfm.complex_matrix_to_real(m) for m in s]), (num_test_samples,8*8))
  s_y = cnot.predict(x=s_real)
  s_m = [np.reshape(ys, (2*dim, 2*dim)) for ys in s_y]
  s_m_pred = [tfm.real_matrix_to_complex(ys) for ys in s_m]
  eigenvalues, traces, eigenvectors = qm.eigen_decomposition([tfm.real_matrix_to_complex(ys) for ys in s_m])
  anti_hermitian_part = [qm.anti_hermitian_part(ys) for ys in s_m_pred]
  anti_hermitian_part_norm = [np.real(np.trace(np.matmul(ys.conj().T, ys))) for ys in anti_hermitian_part]
  print(np.mean(traces))
  print(np.std(traces))
  print(np.mean(anti_hermitian_part_norm))
  print(np.std(anti_hermitian_part_norm))

def compose_circuit():
  epochs = 2000
  num_subepochs = 1
  num_samples = 10000
  units = [64,16,64]
  batch_size = 1000

  hadamard_pi_8th, _ = train_model(
    unitary_transform=np.kron(qm.hadamard, qm.rotate(np.pi/8)),
    name='hadamard_rotate_pi_8th',
    epochs=epochs,
    model=build_non_linear_model(units=units, activation='linear'),
    plot_losses=False,
    num_samples=num_samples,
    num_subepochs=num_subepochs,
    batch_size=batch_size
  )
  cnot, _ = train_model(
    unitary_transform=qm.cnot,
    name='cnot',
    epochs=epochs,
    model=build_non_linear_model(units=units, activation='linear'),
    plot_losses=False,
    num_samples=num_samples,
    num_subepochs=num_subepochs,
    batch_size=batch_size
  )

  num_layers = 2**np.arange(1,16)
  errors = []
  dim = 32

  for layers in num_layers:
    unitary_transform = np.matmul(qm.cnot, np.kron(qm.hadamard, qm.rotate(np.pi/8)))
    unitary_transform = np.linalg.matrix_power(unitary_transform, layers)
    x,y = generate_data(unitary_transform, n=np.shape(unitary_transform)[0], num_samples=10)
    x_cnot = x
    for _ in range(layers):
      x_h_r = hadamard_pi_8th.predict(x=x_cnot)
      x_cnot = cnot.predict(x=x_h_r)
    msq_err = (1.0/num_samples)*np.sum((x_cnot - y)**2)
    # convert back to complex representation
    y = [np.reshape(ys, (2*dim, 2*dim)) for ys in y]
    y = [tfm.real_matrix_to_complex(ys) for ys in y]
    x_cnot = [np.reshape(xs, (2*dim, 2*dim)) for xs in x_cnot]
    x_cnot = [tfm.real_matrix_to_complex(xs) for xs in x_cnot]
    y_0 = qm.partial_trace(y, tensor_dim=[2,2], index=0)
    x_0 = qm.partial_trace(x_cnot, tensor_dim=[2,2], index=0)
    y_0_s = qm.stereographic_projection(qm.bloch_vector(y_0))
    x_0_s = qm.stereographic_projection(qm.bloch_vector(x_0))
    # store the results
    errors.append(msq_err)

  return num_layers, errors

def plot_compose_circuit():
  x,y = compose_circuit()
  fig,axis = plt.subplots(1, 1, figsize=(normal_figure_width, normal_figure_width))
  axis.set_title('Mean squared error vs number of layers')
  axis.set_xscale('log')
  axis.set_yscale('log')
  axis.set_ylabel('error')
  axis.set_xlabel('number of layers')
  axis.plot(x,y)
  fig.savefig('compose_circuit.pdf')


computation_errors = [1.718235700650654e-14, 2.728917872170883e-14, 3.4225502557892086e-14, 1.2396163982650348e-13, 4.996448279934695e-13, 1.8091896260771886e-12, 6.413142939642376e-12, 2.1435518942870975e-11, 9.826842236203848e-11, 6.277303881633471e-10, 2.175825635845086e-09, 6.7780966942311115e-09, 3.627724788342358e-08]

def generate_bottleneck_sweep(name='hadamard', io_dim=16, gate=qm.hadamard, bottleneck_dim = np.arange(1,9)):
  num_samples = 10000
  batch_size = 1000
  epochs = 1000
  num_subepochs = 1
  activation = 'linear'

  epoch_choices = [500,1000,3000]
  loss_results = []
  for epochs in epoch_choices:
    losses = []
    for dim in bottleneck_dim:
      _, result = train_model(
            unitary_transform=gate, 
            epochs=epochs,
            model=build_non_linear_model(units=[io_dim,dim,io_dim], activation=activation),
            plot_losses=False,
            num_samples=num_samples,
            num_subepochs=num_subepochs,
            batch_size=batch_size,
            verbose=0,
      )
      min_loss = np.min(result.training_losses)
      losses.append(min_loss)
    loss_results.append(losses)

  fig, axis = plt.subplots(1, 1, figsize=(normal_figure_width, normal_figure_width))
  # axis.set_title('Loss versus bottleneck dimension')
  axis.set_yscale('log')
  axis.set_xlabel('bottleneck dimension')
  axis.set_ylabel('loss')
  for idx, losses in enumerate(loss_results):
    axis.plot(bottleneck_dim, losses, label='epochs = {}'.format(epoch_choices[idx]), marker='.')
  axis.legend()
  axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
  fig.tight_layout()
  fig.savefig('bottle_neck_sweep_{}.pdf'.format(name))


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
  results = []
  for _ in range(num_runs):
    _, result = train_model(
        unitary_transform=gate, 
        name='density matrix - hadamard gate',
        epochs=epochs,
        model=build_non_linear_model(units=units, activation=activation),
        plot_losses=False,
        num_samples=num_samples,
        num_subepochs=num_subepochs,
        batch_size=batch_size
      )
    results.append(result)
  return results

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
  results = generate_loss_sweep(
    gate, 
    units=units, 
    batch_size=batch_size, 
    num_runs=num_runs, 
    epochs=epochs, 
    num_subepochs=num_subepochs
  )
  pickle.dump(results, open('results/sweep_{}_{}_{}.p'.format(batch_size, num_runs, epochs) , 'wb'))
  linestyle = ['solid', 'dashed', 'dashdot', 'dotted']

  fig, axis = plt.subplots(3, 1, figsize=(normal_figure_width,3*0.5*normal_figure_width))
  # axis[0].set_title('Training progress')
  axis[0].set_yscale('log')
  axis[0].set_xlabel('steps')
  axis[0].set_ylabel('loss')
  for r in results:
    axis[0].plot(r.training_losses)

  steps = int(epochs/num_subepochs)*np.arange(0,num_subepochs)

  axis[1].set_title('Average trace')
  axis[1].set_xlabel('steps')
  axis[1].set_ylabel('trace')
  for r in results:
    axis[1].errorbar(x=steps, y=r.mean_traces, yerr=r.std_traces)

  axis[2].set_title('Average norm anti-hermitian part')
  axis[2].set_xlabel('steps')
  axis[2].set_ylabel('norm')
  axis[2].set_yscale('log')

  for idx, r in enumerate(results):
    axis[2].errorbar(x=steps, y=r.mean_anti_hermitian_parts, yerr=r.std_anti_hermitian_parts, linestyle=linestyle[idx])

  fig.tight_layout()
  fig.savefig('sweep_{}.pdf'.format(name))
  return results

def generate_plots():
  # 
  #print("Hadamard")
  #generate_loss_sweep_plot(title='', gate=qm.hadamard, units= [16,3,16], epochs=3000, num_subepochs=300)
  #
  print("CNOT")
  generate_loss_sweep_plot(name='cnot', gate=qm.cnot, units=[64,15,64], epochs=1000, num_subepochs=100, num_runs=1)
  # 
  print("Bottleneck Sweep")
  generate_bottleneck_sweep(name='cnot', io_dim=64, gate=qm.cnot, bottleneck_dim=np.arange(12,20))
  