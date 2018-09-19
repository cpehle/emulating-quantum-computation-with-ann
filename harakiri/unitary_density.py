import numpy as np
import pickle
import multiprocessing as mp

from pathlib import Path

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

def generate_unnormalized_data(u=qm.hadamard, n=2, num_samples=10000, use_uniform_samples=False):
  """
  Args:
    u (quantum gate): quantum gate.
    n (int): Dimension of the quantum gate.
    num_samples (int): Number of samples.
    use_uniform_samples (bool): Whether to use uniform or ginibre samples.
  """
  if use_uniform_samples:
    x,x_density = rnd.uniform_density_samples(dim=n, num_samples=num_samples)
  else:
    x,x_density = rnd.ginibre_density_samples(dim=n, num_samples=num_samples)
  y = np.matmul(np.matmul(u, x_density), u.conj().T)
  x_real = np.reshape(np.stack([tfm.complex_matrix_to_real(m) for m in x]), (num_samples,(2*n)**2))
  y_real = np.reshape(np.stack([tfm.complex_matrix_to_real(m) for m in y]), (num_samples,(2*n)**2))
  return x_real, y_real

def generate_data(u=qm.hadamard, n=2, num_samples=10000, use_uniform_samples=False):
  """
  Args:
    u (quantum gate): quantum gate.
    n (int): Dimension of the quantum gate.
    num_samples (int): Number of samples.
    use_uniform_samples (bool): Whether to use uniform or ginibre samples.
  """
  if use_uniform_samples:
    x,x_density = rnd.uniform_density_samples(dim=n, num_samples=num_samples)
  else:
    x,x_density = rnd.ginibre_density_samples(dim=n, num_samples=num_samples)

  y = np.matmul(np.matmul(u, x_density), u.conj().T)
  x_real = np.reshape(np.stack([tfm.complex_matrix_to_real(m) for m in x_density]), (num_samples,(2*n)**2))
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
    self.min_anti_hermitian_parts = []
    self.max_anti_hermitian_parts = []
    self.max_traces = []
    self.min_traces = []

def analysis(result, y_pred, dim):
  """
  """
  y_pred = [np.reshape(ys, (2*dim, 2*dim)) for ys in y_pred]
  y_pred = [tfm.real_matrix_to_complex(ys) for ys in y_pred]
  eigenvalues, traces, eigenvectors = qm.eigen_decomposition(y_pred)
  hermitian_part = [qm.hermitian_part(ys) for ys in y_pred]
  hermitian_part_norm = [np.real(np.trace(np.matmul(ys.conj().T, ys))) for ys in hermitian_part]
  anti_hermitian_part = [qm.anti_hermitian_part(ys) for ys in y_pred]
  anti_hermitian_part_norm = [np.real(np.trace(np.matmul(ys.conj().T, ys))) for ys in anti_hermitian_part]
  result.mean_traces.append(np.mean(traces))
  result.std_traces.append(np.std(traces))
  result.min_traces.append(np.min(traces))
  result.max_traces.append(np.max(traces))
  result.mean_hermitian_parts.append(np.mean(hermitian_part_norm))
  result.std_hermitian_parts.append(np.std(hermitian_part_norm))
  result.mean_anti_hermitian_parts.append(np.mean(anti_hermitian_part_norm))
  result.min_anti_hermitian_parts.append(np.min(anti_hermitian_part_norm))
  result.max_anti_hermitian_parts.append(np.max(anti_hermitian_part_norm))
  result.std_anti_hermitian_parts.append(np.std(anti_hermitian_part_norm))

def train_model(
  unitary_transform = qm.hadamard, 
  name='',
  epochs=3000, 
  num_samples=10000,
  batch_size=1000,
  model=build_non_linear_model(units=[16,32,32,16]),
  plot_losses=True,
  use_unnormalized_data=False,
  use_uniform_samples=False,
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
    if use_unnormalized_data:
      x,y = generate_unnormalized_data(
        unitary_transform, 
        n=np.shape(unitary_transform)[0], 
        num_samples=num_samples, 
        use_uniform_samples=use_uniform_samples
      )
    else:
      x,y = generate_data(
        unitary_transform, 
        n=np.shape(unitary_transform)[0], 
        num_samples=num_samples, 
        use_uniform_samples=use_uniform_samples
      )
  else:
    x,y = data
  dim = np.shape(unitary_transform)[0]

  callbacks = []
  result = TrainingResult()
  sub_epochs = int(epochs/num_subepochs)

  for _ in range(num_subepochs):
    history = model.fit(
      x=x,
      y=y,
      batch_size=batch_size,
      epochs=sub_epochs,
      validation_split=0.05,
      callbacks=callbacks,
      verbose=verbose
    )
    y_pred = model.predict(x=x)
    analysis(result, y_pred, dim)

    result.training_losses += history.history['loss']
    result.validation_losses += history.history['val_loss']

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

def quantumness(use_uniform_samples = False):
  epochs = 3000
  num_subepochs = 1
  num_samples = 10000
  
  dimensions = np.arange(12,21)
  batch_size = 1000

  result = TrainingResult()

  traces_result = []
  traces_std_result = []
  anti_hermitian_part_norm_result = []
  anti_hermitian_part_norm_std_result = []

  # test data is held constant over the different bottleneck dimensions
  num_test_samples = 10000
  dim = 4
  if use_uniform_samples:
    s = np.stack([rnd.density_matrix_uniform(n = 4) for _ in range(num_test_samples)])
    s_real = np.reshape(np.stack([tfm.complex_matrix_to_real(m) for m in s]), (num_test_samples,8*8))
  else:
    s = np.stack([rnd.ginibre_ensemble_sample(n = 4) for _ in range(num_test_samples)])
    s_real = np.reshape(np.stack([tfm.complex_matrix_to_real(m) for m in s]), (num_test_samples,8*8))

  for d in dimensions:
    units = [64,d,64]
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
    s_y = cnot.predict(x=s_real)
    analysis(result, s_y, dim)

    s_m = [np.reshape(ys, (2*dim, 2*dim)) for ys in s_y]
    s_m_pred = [tfm.real_matrix_to_complex(ys) for ys in s_m]
    eigenvalues, traces, eigenvectors = qm.eigen_decomposition([tfm.real_matrix_to_complex(ys) for ys in s_m])
    anti_hermitian_part = [qm.anti_hermitian_part(ys) for ys in s_m_pred]
    anti_hermitian_part_norm = [np.real(np.trace(np.matmul(ys.conj().T, ys))) for ys in anti_hermitian_part]
    traces_res = np.abs(np.real(1 - traces))
    mean_traces = np.mean(traces_res)
    traces_result.append(mean_traces)
    result.mean_traces.append(mean_traces)
    result.std_traces.append(np.std(traces_res))

    traces_std_result.append(np.std(traces_res))
    anti_hermitian_part_norm_result.append(np.mean(anti_hermitian_part_norm))
    anti_hermitian_part_norm_std_result.append(np.std(anti_hermitian_part_norm))

  fig, axis = plt.subplots(1, 1, figsize=(normal_figure_width, normal_figure_width))
  axis.set_yscale('log')
  axis.set_xlabel('bottleneck dimension')
  axis.set_ylabel('loss')  
  linestyle = ['solid', 'dashed', 'dashdot', 'dotted']
  axis.plot(dimensions, traces_result, label='mean residual trace', linestyle=linestyle[0])
  axis.plot(dimensions, anti_hermitian_part_norm_result, label='mean anti-herm. norm', linestyle=linestyle[1])
  axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
  axis.legend()
  fig.tight_layout()
  label = 'uniform' if use_uniform_samples else 'ginibre'
  fig.savefig('quantumness_{}.pdf'.format(label))
  return

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
  dim = 4

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
    #y_0 = qm.partial_trace(y, tensor_dim=[2,2], index=0)
    #x_0 = qm.partial_trace(x_cnot, tensor_dim=[2,2], index=0)
    #y_0_s = qm.stereographic_projection(qm.bloch_vector(y_0))
    #x_0_s = qm.stereographic_projection(qm.bloch_vector(x_0))
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
  fig.tight_layout()
  fig.savefig(Path('compose_circuit.pdf'))

def generate_bottleneck_sweep(
    name='hadamard', 
    io_dim=16, 
    gate=qm.hadamard, 
    bottleneck_dim=np.arange(1,9), 
    use_unnormalized_data=False,
    use_uniform_samples=False,
    ):
  num_samples = 10000
  batch_size = 1000
  epochs = 1000
  num_subepochs = 1
  activation = 'linear'

  epoch_choices = [1000,3000,10000]
  loss_results = []
  for epochs in epoch_choices:
    losses = []
    for dim in bottleneck_dim:
      print("training bottleneck_dim {} epochs {}".format(dim, epochs))
      _, result = train_model(
            unitary_transform=gate, 
            epochs=epochs,
            model=build_non_linear_model(units=[io_dim,dim,io_dim], activation=activation),
            plot_losses=False,
            num_samples=num_samples,
            num_subepochs=num_subepochs,
            batch_size=batch_size,
            verbose=0,
            use_unnormalized_data=use_unnormalized_data,
            use_uniform_samples=use_uniform_samples
      )
      min_loss = np.min(result.training_losses)
      losses.append(min_loss)
    loss_results.append(losses)

  fig, axis = plt.subplots(1, 1, figsize=(normal_figure_width, normal_figure_width))
  fig.tight_layout()
  axis.set_yscale('log')
  axis.set_xlabel('bottleneck dimension')
  axis.set_ylabel('loss')
  axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

  for idx, losses in enumerate(loss_results):
    axis.plot(bottleneck_dim, losses, label='epochs = {}'.format(epoch_choices[idx]))
 
  axis.legend()
  fig.savefig('bottle_neck_sweep_{}.pdf'.format(name))

def generate_loss_sweep(
    gate = qm.hadamard,
    units = [16,3,16],
    num_runs = 100,
    num_samples = 10000,
    batch_size = 1000,
    epochs = 300,
    num_subepochs = 100,
    activation = 'linear',
    use_unnormalized_data = False,
    use_uniform_samples = False,
    verbose = True
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
        batch_size=batch_size,
        use_unnormalized_data=use_unnormalized_data,
        use_uniform_samples=use_uniform_samples,
        verbose=verbose,
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
    use_unnormalized_data = False,
    use_uniform_samples=False,
    verbose=True,
  ):
  results = generate_loss_sweep(
    gate, 
    units=units, 
    batch_size=batch_size, 
    num_runs=num_runs, 
    epochs=epochs, 
    num_subepochs=num_subepochs,
    use_unnormalized_data=use_unnormalized_data,
    use_uniform_samples=use_uniform_samples,
    verbose=verbose
  )
  pickle.dump(results, open('results/sweep_{}_{}_{}.p'.format(batch_size, num_runs, epochs) , 'wb'))

  linestyle = ['solid', 'dashed', 'dashdot', 'dotted']
  fig, axis = plt.subplots(3, 1, sharex=True, figsize=(normal_figure_width,5*0.5*normal_figure_width))
  axis[0].set_yscale('log')
  axis[0].set_ylabel('loss')
  for r in results:
    axis[0].plot(r.training_losses, linestyle='solid')

  steps = int(epochs/num_subepochs)*np.arange(0,num_subepochs)

  axis[1].set_ylabel('trace')
  axis[1].set_yscale('log')

  for idx,r in enumerate(results):
    axis[1].plot(steps, r.mean_traces, label="mean", linestyle='solid')
    axis[1].plot(steps, r.min_traces, label="min", linestyle='dashed')
    axis[1].plot(steps, r.max_traces, label="max", linestyle='dotted')
    if idx is 0:
      axis[1].legend()

  axis[2].set_ylabel('norm anti-hermitian part')
  axis[2].set_yscale('log')

  for idx, r in enumerate(results):
    axis[2].plot(steps, r.mean_anti_hermitian_parts, label="mean", linestyle='solid')
    axis[2].plot(steps, r.min_anti_hermitian_parts, label="min", linestyle='dashed')
    axis[2].plot(steps, r.max_anti_hermitian_parts, label="max", linestyle='dotted')
    if idx is 0:
      axis[2].legend()

  axis[2].set_xlabel('epoch')
  for ax in axis:
    ax.label_outer()

  fig.subplots_adjust(hspace=0)
  fig.tight_layout()
  fig.savefig('sweep_{}.pdf'.format(name))
  return results

def generate_plots():
  for use_uniform_samples in [True, False]:
    label = "uniform" if use_uniform_samples else "ginibre"
    for dim in [15,16]:
      print('cnot_{}_{}'.format(dim, label))
      generate_loss_sweep_plot(
        name='cnot_{}_{}'.format(dim, label), 
        gate=qm.cnot, 
        units=[64,dim,64], 
        epochs=1000, 
        num_subepochs=100, 
        num_runs=1, 
        use_uniform_samples=use_uniform_samples,
        verbose=False
      )
      print('cnot_unnormalized_{}_{}'.format(dim, label))
      generate_loss_sweep_plot(
        name='cnot_unnormalized_{}_{}'.format(dim, label), 
        gate=qm.cnot, 
        units=[64,dim,64], 
        epochs=1000, 
        num_subepochs=100, 
        num_runs=1, 
        use_unnormalized_data=True, 
        use_uniform_samples=use_uniform_samples,
        verbose=False
      )
    print("bottleneck sweep cnot {}".format(label))
    generate_bottleneck_sweep(
      name='cnot_{}'.format(label), 
      io_dim=64, 
      gate=qm.cnot, 
      bottleneck_dim=np.arange(12,20),
      use_uniform_samples=use_uniform_samples,
    )
    print("bottleneck sweep cnot unnormalized {}".format(label))
    generate_bottleneck_sweep(
      name='cnot_unnormalized_{}'.format(label), 
      io_dim=64, 
      gate=qm.cnot, 
      bottleneck_dim=np.arange(12,20), 
      use_unnormalized_data=True,
      use_uniform_samples=use_uniform_samples,
    )
  print("quantumness")
  quantumness()

if __name__ == '__main__':
  generate_plots()
