import numpy as np

def real_of_complex(z):
  """
  Convert a n-dimensional complex vector into a 2n-dimensional
  real vector.

  Args:
    z: numpy float tensor of shape (n,2)
  """
  return np.vstack((np.real(z[:,0]),np.imag(z[:,0]),np.real(z[:,1]),np.imag(z[:,1]))).T


def real_of_complex_tensor(z):
  """
  """
  x = np.real(z)
  y = np.imag(z)
  return np.hstack((x,y))

def complex_of_real_tensor(x):
  """
  """
  x,y = np.hsplit(x, 2)
  return np.complex(x,y)


def real_of_complex_test():
  z = np.array([[1.0 + 1.0j, 2.0 + 2.0j]])
  x_ = np.array([[1.0, 1.0, 2.0, 2.0]])
  x = real_of_complex(z)
  np.testing.assert_array_almost_equal(x, x_)

def complex_of_real_transposed(y):
  """
  """
  return np.vstack((np.complex(y[:,0],y[:,1]), np.complex(y[:,2],y[:,3]))).T

def complex_of_real(y):
  """
  """
  return np.vstack((y[:,0] + 1.0j*y[:,1], y[:,2] + 1.0j*y[:,3]))

def cartesian_of_projective(z):
  """
  Computes the cartesian coordinates for projective coordinates
  [z_0 : z_1] in CP^1.
  """
  alpha = z[0,:]
  beta = z[1,:]
  vz = np.real(np.conj(alpha)*alpha - np.conj(beta)*beta)
  vx_plus_ivy = 2*np.conj(alpha)*beta
  vx = np.real(vx_plus_ivy)
  vy = np.imag(vx_plus_ivy)
  return vx, vy, vz

def theta_of_complex(z):
  """
  Computes the angle theta on the Bloch sphere.
  """
  return 2*np.arcsin(np.sqrt(np.real((z * np.conj(z)))))

def phi_of_complex(z):
  """
  Computes the angle phi on the bloch sphere.
  """
  theta = theta_of_complex(z)
  return np.arccos(np.real(z/np.sin(theta/2.0)))

def theta_phi_of_complex(z):
  """
  Computes both theta and phi on the Bloch sphere given
  a tensor of shape (2,n) and returns the as a tensor of 
  shape (2,n)
  """
  return np.stack([theta_of_complex(z), phi_of_complex(z)], axis=1)