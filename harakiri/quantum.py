import numpy as np

# the sigma matrices
sigma_0 = np.array([[1.0, 0.0], 
                    [0.0, 1.0]])
sigma_1 = np.array([[0.0, 1.0], 
                    [1.0, 0.0]])
sigma_2 = np.array([[0.0, -1.0j], 
                    [1.0j, 0.0]])
sigma_3 = np.array([[1.0, 0.0], 
                    [0.0,-1.0]])

# |0> and |1> states in the sigma_3 eigen basis
zero = np.array([[1.0 + 0.0j],
                 [0.0 + 0.0j]])
one = np.array([[0.0 + 0.0j],
                [1.0 + 0.0j]])

# hadamard gate
hadamard = 1.0/np.sqrt(2) * (sigma_1 + sigma_3)

def psi(theta, phi):
  """
  Defines states on bloch sphere given by set of angles. 

  Args:
    theta np.array(float): Theta angle with the z-axis of the bloch sphere.
    phi np.array(float): Phi angle in  the xy-plane of the bloch sphere.

  Returns: 
    psi np.array(float)
  """
  return np.cos(theta/2) * zero + np.exp(1.0j*phi) * np.sin(theta/2) * one

def normalize_state(psi):
  """
  Normalize a vector of states.

  Args:
    psi: State vectors to be normalized.
  Returns:
    Normalized state vector.
  """
  norm = np.sqrt(np.real(np.dot(psi.T, psi)))
  return psi/norm

def norm(psi):
  """
  Compute the norm of the state `psi`.
  """
  return np.sqrt(np.real(np.dot(psi.T, psi)))

def nkron(*args):
  """
  Calculate a kronecker product over a variable number of inputs.

  Args:
    args: Matrices / vectors of which the tensor product is to be formed.
  """
  result = np.array([[1.0]])
  for op in args:
    result = np.kron(result, op)
  return result

p_0 = np.dot(zero, zero.T)
p_1 = np.dot(one, one.T)

cnot = np.kron(p_0,sigma_0) + np.kron(p_1,sigma_1)

def rotate(phi):
  """
  Implements the R(phi) rotation gate.

  Args:
    phi (float): Rotation angle.
  """
  return np.array([[1.0, 0.0],
                   [0.0, np.exp(1.0j*phi)]])

def rho(psi):
  """
  Compute the density of state matrix.

  Args:
    psi (tensor of shape (2,n))
  """
  return np.dot(psi, psi.T)