import numpy as np

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
