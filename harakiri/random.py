import numpy as np

def unitary_haar_measure(n):
  """
  A random unitary matrix distributed with Haar measure
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
  """
  p = sphere(2*n+1,m)
  x = p[:,0:n]
  y = p[:,(n+1):(2*n+1)]
  z = x + 1.0j*y
  return z
