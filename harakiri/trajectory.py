import numpy as np

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

def generate_sin_data(periods = 4.5, n = 1000, split = 0.8):
  """Generate data for a sine curve.

  Args:
    periods (float): Number of sine periods to use.
    n (int): Number of sample points to create.
    split (float): Split between the training and validation set.
  """
  t = np.mgrid[0.0:periods*2.0*np.pi:1000j]
  f = np.sin(t)
  k = int(split * n)

  f_train = f[:k]
  f_test = f[k+1:]

  x_train = f_train[0:k-2]
  y_train = f_train[1:k-1]
  x_test = f_test[:-1]
  y_test = f_test[:]

  # keras expects the input to an LSTM network to have this form.
  x_train = np.reshape(x_train, (np.shape(x_train)[0],1,1))
  y_train = np.reshape(y_train, (np.shape(y_train)[0],1))
  x_test = np.reshape(x_test, (np.shape(x_test)[0],1,1))
  y_test = np.reshape(y_test, (np.shape(y_test)[0],1))
  return x_train, y_train, x_test, y_test


def convert_output_to_real(y):
  # TODO(Christian): Delete this.
  return np.vstack((np.real(y[:,0]),np.imag(y[:,0]),np.real(y[:,1]),np.imag(y[:,1]))).T

def generate_bloch_sphere_trajectory(psi_initial, one_parameter_subgroup, steps=10000):
  """Generate a bloch sphere trajectory based on a one-parameter subgroup.

  Note: This should be extended to the case of piecewise one-parameter subgroups.

  Args:
    psi_initial: Initial bloch sphere state.
    one_parameter_subgroup: One parameter subgroup of unitary transformations to be used.
    steps: number of timesteps to use. TODO(Christian): This is not actually implemented.
  """
  t = np.mgrid[0.0:8:10000j]
  U = np.moveaxis(one_parameter_subgroup(t),2,0)
  psi = np.matmul(U,psi_initial)
  psi_real = convert_output_to_real(psi)
  x = psi_real[0:steps-2]
  y = psi_real[1:steps-1]
  return np.reshape(x, (np.shape(x)[0],1,4)),np.reshape(y, (np.shape(y)[0],4))

def build_lstm_model(layers, optimizer='rmsprop'):
  """Create a simple time series prediction model.

  Args:
    layers (list): List of integers for specifying the dimensionality
                   of the three layers.
  """
  model = Sequential()
  model.add(LSTM(
      units=layers[1], 
      input_shape=(None, layers[0]),
      return_sequences=True
  ))  
  model.add(Dropout(0.2))
  model.add(LSTM(
      units=layers[2],
      return_sequences=False
  ))
  model.add(Dropout(0.2))
  model.add(Dense(
      units=layers[3]
  ))
  model.add(Activation("linear"))
  model.compile(loss="mse", optimizer=optimizer)
  return model

def train_1d_sin_model(epochs=500, plot_losses=True):
  """Train a 1d sin model.

  Args:
    epochs (int): number of epochs of training to use.
  """
  x_train, y_train, x_test, y_test = generate_sin_data()
  model = build_lstm_model([1, 50, 100, 1])
  history = model.fit(
      x_train,
      y_train,
      verbose=0,
      batch_size=512,
      epochs=epochs,
      validation_split=0.05
  )

  if plot_losses:
    import matplotlib.pyplot as plt

    # TODO(Christian): Put the two plots in one.
    plt.plot(x_train[:,0,0], label='training')
    plt.plot(model.predict(x_train), label='predicted')
    plt.legend()
    plt.show()

    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    plt.plot(training_loss, label='training loss')
    plt.plot(validation_loss, label='validation loss')
    plt.yscale('log')
    plt.xlabel('steps')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
