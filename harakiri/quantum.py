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
hadamard = 1.0/np.sqrt(2) * (sigma_1 + sigma_3) + 1.0j*np.zeros((2,2))

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
  """Compute the norm of the state `psi`.
  """
  return np.sqrt(np.real(np.dot(psi.T, psi)))

def nkron(*args):
  """Calculate a kronecker product over a variable number of inputs.

  Args:
    args: Matrices / vectors of which the tensor product is to be formed.
  """
  result = np.array([[1.0]])
  for op in args:
    result = np.kron(result, op)
  return result

p_0 = np.dot(zero, zero.T)
p_1 = np.dot(one, one.T)

cnot = (np.kron(p_0,sigma_0) + np.kron(p_1,sigma_1))

def rotate(phi):
  """Implements the R(phi) rotation gate.

  Args:
    phi (float): Rotation angle.
  """
  return np.array([[1.0, 0.0],
                   [0.0, np.exp(1.0j*phi)]])

def rho(psi):
  """Compute the density of state matrix for a pure state.

  Args:
    psi (tensor of shape (2,n))
  """
  return np.dot(psi, psi.conj().T)

def measure():
  """Measure an n-Qbit state. This is non-deterministic.
  """
  pass

def anti_hermitian_part(m):
  return 1/2*(m - m.conj().T)

def hermitian_part(m):
  return 1/2*(m + m.conj().T)

def eigen_decomposition(unitaries):
  """
  Compute the eigen values and vectors of an array of (unitary) matrices.

  Args:
    unitaries (array): Array of unitary matrices.
  """
  eigenvalues, eigenvectors = np.linalg.eig(unitaries)
  traces = np.sum(eigenvalues, axis=1)
  return eigenvalues, traces, eigenvectors

def partial_trace(matrix, tensor_dim=[2,2], index=0):
  shape = tensor_dim + tensor_dim
  tensor = matrix.reshape(shape)
  return np.trace(tensor, axis1=index, axis2=index+len(tensor_dim))

def bloch_vector(rho):
  """
  Compute the bloch vector from a given vector of 2-state density matrix.
  """
  u = 2*np.real(rho[:,0,1])
  v = 2*np.imag(rho[:,1,0])
  w = rho[:,0,0] - rho[:,1,1]
  return np.vstack([u,v,w]).T

def stereographic_projection(coordinates):
  u = np.real(coordinates)[:,0]
  v = np.real(coordinates)[:,1]
  w = np.real(coordinates)[:,2]
  w_inv = 1/(1 - w)
  x = w_inv * u
  y = w_inv * v
  return np.vstack([x,y]).T
