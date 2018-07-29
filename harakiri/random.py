import numpy as np
import matplotlib.pyplot as plt

def unitary_haar_measure(n):
  """
  A random unitary matrix distributed with Haar measure.

  Args:
    n (int): Dimension of the unitary group U(n).
  """
  z = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2.0)
  q,r = np.linalg.qr(z)
  d = np.diagonal(r)
  ph = d/np.absolute(d)
  q = np.multiply(q,ph,q)
  return q

def sphere_rejection(n, m):
  """
  Generate m samples uniformly distributed on the n-Sphere
  using the rejection method.

  Args:
    n (int): Dimension of the sphere S^n.
    m (int): Number of samples to be generated.
  """
  j = 0
  y = np.zeros((m,n))
  while j < m:
    x = 2 * np.random.rand(n)-1
    l = np.linalg.norm(x)
    if (l <= 1):
      y[j,:] = 1/l * x
      j=j+1
  return y

def complex_projective_spherical(n,m,sphere = sphere_rejection):
  """
  Generate m random samples in complex projective space CP^n using 
  a sphere sampling method.

  Args:
    n (int): Complex dimension of the complex projective space CP^n.
    m (int): Number of samples to be generated.
    sphere: Sphere sampling method to be used.
  """
  p = sphere(2*n+1,m)
  x = p[:,0:n]
  y = p[:,(n+1):(2*n+1)]
  z = x + 1.0j*y
  return z

# Functions specific to the bloch sphere
def random_angles_np(size):
  return 2*np.pi*np.random.rand(size)

def random_angles_within(low, high, size):
  return (np.random.rand(size) / (high - low)) + low

def random_theta_phi(size):
  return np.stack([random_angles_np(size), random_angles_np(size)], axis=1)

def theta_phi_within(size, phi_low, phi_high, theta_low, theta_high):
  return np.stack([random_angles_within(theta_low, theta_high, size), 
                   random_angles_within(phi_low, phi_high, size)], axis=1)

def uniform_2d_spherical(m):
  """
  Generate random uniform samples on a 2d-sphere.

  Args:
    m (int): Number of samples to generate.
  """
  phi =  2 * np.pi * np.random.rand(m)
  theta = np.arccos(1 - 2 * np.random.rand(m))
  return phi,theta

def uniform_2d_spherical_within(
    m, 
    phi_min = 0.0, 
    phi_max=2.0*np.pi, 
    theta_min=0.0, 
    theta_max=np.pi):
  """
  Generate random uniform samples on a 2d-sphere within the 
  specified spherical coordinates (phi_min, phi_max) and
  (theta_min, theta_max).

  Args:
    m (int): Number of samples to generate
    phi_min (float): Minimum angle to use
    phi_max (float): Maximum angle to use
    theta_min (float): Minimum angle to use
    theta_max (float): Maximum angle to use
  """
  delta_phi = phi_max - phi_min
  delta_theta = theta_max - theta_min
  phi = delta_phi * np.random.rand(m) + phi_min
  theta = (delta_theta/np.pi) * np.arccos(1 - 2 * np.random.rand(m)) + theta_min
  return phi,theta


def ginibre_ensemble_sample(n = 2):
  """Generate random matrix in  C^{n x n} based
  on $2n x 2n $ independent random uniformly drawn samples.

  Args:
    n (int): Dimension of the space of matrices.
  """
  return np.random.randn(n,n) + 1.0j*np.random.randn(n,n)

def density_matrix_ginibre_sample(n = 2):
  """Generate random desnsity matrix based on the ginibre ensemble.

  Args:
    n (int): Dimension of the space of matrices
  """
  s = ginibre_ensemble_sample(n = n)
  h = np.matmul(s, s.conj().T)
  h_trace = np.trace(h)
  return h / h_trace

def density_matrix_burres_sample(n = 2):
  """Generate a random density matrix based on the burres metric
  
  See https://arxiv.org/pdf/1010.3570.pdf c.f. equation 22

  Args:
    n (int): Dimension of the space of matrices
  """
  s = ginibre_ensemble_sample(n = n)
  u = unitary_haar_measure(n = n)
  f = (np.eye(n) + u)
  f_conj = f.T.conj()
  g = np.matmul(s, s.conj().T)
  m = np.matmul(np.matmul(f, g), f_conj)
  return m/np.trace(m)


def density_matrix_ginibre(n = 2, m = 1000):
  """Generate random desnsity matrices based on the ginibre ensemble.

  Args:
    n (int): Dimension of the space of matrices
    m (int): Number of samples
  """
  return np.stack([density_matrix_ginibre_sample(n = n) for i in range(m)])

def plot_ginibre_ensemble_eigenvalues(n=4, m=100000, bins=1000):
  """
  """
  matrices = [ginibre_ensemble_sample(n) for _ in range(m)]
  eigenvalues = np.reshape(np.linalg.eigvals(matrices), n*m)
  x, y = np.real(eigenvalues), np.imag(eigenvalues)
  plt.hist2d(x,y,bins=bins)

def plot_density_matrix_eigenvalues(n = 2, m = 1000, bins = 1000):
  """
  """
  matrices = density_matrix_ginibre(n = n, m = m)
  eigenvalues = np.reshape(np.linalg.eigvals(matrices), n*m)
  x, y = np.real(eigenvalues), np.imag(eigenvalues)
  plt.hist2d(x,y,bins=bins)

def plot_eigenvalues(m, bins=1000):
  """
  """
  a,b,_ = np.shape(m)
  eigenvalues = np.reshape(np.linalg.eigvals(m), a*b)
  x, y = np.real(eigenvalues), np.imag(eigenvalues)
  plt.hist2d(x,y,bins=bins)
  return x,y


def probability_simplex_samples(d, n = 1000):
  """
  """
  def sample(d):
    dist = -np.log(np.random.rand(d))
    norm = np.sum(dist)
    return dist/norm
  return np.array([sample(d) for _ in range(n)])


def generate_diagonal_probability_matrices(d,n):
  """
  """
  return [np.diag(v) for v in probability_simplex_samples(d, n)]

def generate_density_matrices_unitary(d, n):
  m = generate_diagonal_probability_matrices(d=d, n=n)
  u = unitary_haar_measure(d)
  return np.matmul(u,m)

