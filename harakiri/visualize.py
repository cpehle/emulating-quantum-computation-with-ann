import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import transform as tfm
from . import quantum as qm

def sphere_points_2d():
  """ 
  Generate points on the sphere S^2 in an grid.

  Returns:
    x,y,z (tuple): cartesian coordinates of the points that were generated.
  """
  r = 1
  phi,theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
  x = r * np.sin(phi) * np.cos(theta)
  y = r * np. sin(phi) * np.sin(theta)
  z = r * np.cos(phi)
  return x,y,z

def bloch_sphere():
  """
  Create a plot of the bloch sphere.
  """
  x,y,z = sphere_points_2d()
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(
    x, y, z,  rstride=1, cstride=1, color='c', alpha=0.3, linewidth=0)
  ax.set_xlim([-1,1])
  ax.set_ylim([-1,1])
  ax.set_zlim([-1,1])
  ax.set_aspect("equal")
  plt.tight_layout()
  return plt, ax

def spherical2d(ax, theta, phi, color='read', label=''):
  pass

def projective(ax, psi, color='red', label=''):
  """
  Create a scatter plot of points in CP^1 on a given figure.

  Args:
    ax: Matplotlib figure to use.
    psi: Points in CP^1.
    color (str): Color to use for the scatter plot.
    label (str): Label to use for the scatter plot.
  """
  x, y, z = tfm.cartesian_of_projective(psi)
  ax.scatter(xs=x, ys=y, zs=z, color=color, s=20, alpha=1.0, label=label)

def psi(psi, color='red'):
  """
  Plot an array of points in projective space CP^1.
  """
  plt, ax = bloch_sphere()
  projective(ax, psi=psi, color=color)
  plt.show()

def phi_theta(theta, phi, color='red'):
  """
  Plot array of theta and phi.
  """
  psi = qm.psi(theta=theta, phi=phi)
  psi(psi, color=color)