import numpy as np

def grad(w, f, noise):
  """Compute a gradient based on samples from the fitness function.
  """
  f -= f.mean()
  f /= f.std() # standardize the rewards to be N(0,1) gaussian
  g = np.dot(f, noise)
  return g

def parameter_proposal(w, k = 200, sigma = 3):
  """Generate a new set of parameter proposals.
  """
  noise = np.random.randn(k, *np.shape(w))
  return (np.expand_dims(w, 0) + sigma * noise, noise)

def update(w, g, alpha = 0.03):
  """Perfom a parameter update step.
  """
  return w + alpha * g

def round_stochastic(w):
  """Round stochastically to the nearest value.
  """
  w_ = np.floor(w)
  ps = w - w_ 
  return w_ + np.random.binomial(1, p=ps)
