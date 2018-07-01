import numpy as np

def real_of_complex(y):
  """
  """
  return np.vstack((np.real(y[:,0]),np.imag(y[:,0]),np.real(y[:,1]),np.imag(y[:,1]))).T

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
  """
  return 2*np.arcsin(np.sqrt(np.real((z * np.conj(z)))))

def phi_of_complex(z):
  """
  """
  theta = theta_of_complex(z)
  return np.arccos(np.real(z/np.sin(theta/2.0)))

def theta_phi_of_complex(z):
  """
  """
  return np.stack([theta_of_complex(z), phi_of_complex(z)], axis=1)