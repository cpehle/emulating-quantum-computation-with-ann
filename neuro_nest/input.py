import numpy as np

class SignedSpike(object):
    def __init__(self, source, sign, timestep):
        self.source = source
        self.sign = sign
        self.timestep = timestep

def spike_tensor_quantization(v, T):
    """Quantize a vector in R^d into a sequence of signed events.
    c.f. Algorithm 4 in https://arxiv.org/pdf/1602.08323.pdf.

    The sequence will converge with rate 1/T.

    Args:
        v: numpy float tensor of arbitrary shape to be quantized.
        T: total number of timesteps to use.
    """
    phi = np.zeros_like(v)

    for t in range(T):
        phi += v
        phi_abs = np.abs(phi)
        while np.max(phi_abs) > 0.5:
            i = np.argmax(phi_abs)
            s = np.sign(phi[i])
            phi[i] -= s
            yield SignedSpike(source=i, sign=s, timestep=t)


def stochastic_vector_sampling(v, T):
    """Stochastically sample from a vector in R^d.
    c.f. Algorithm 5 in https://arxiv.org/pdf/1602.08323.pdf.

    The sequence will converge with a rate 1/sqrt(T).

    Args:
        v: numpy float tensor of arbitrary shape to be quantized.
        T: total number of timesteps to use.
    """
    mag = np.sum(np.abs(v))
    p = np.abs(v)/mag
    generated_spikes = []

    for t in range(T):
        N = np.random.poisson(mag) # total number of events to be generated
        for _ in range(N):
            i = np.random.multinomial(1, pvals=p)
            s = np.sign(v[i])
            yield SignedSpike(source=i, sign=s, timestep=t)

def spike_tensor_stream_quantization(v, timesteps):
    """Quantize a vector in R^d into a sequence of signed events.
    c.f. Algorithm 6 in https://arxiv.org/pdf/1602.08323.pdf.

    The sequence will converge with rate 1/T.

    Args:
        v: numpy float tensor of arbitrary shape to be quantized.
        timesteps: total number of timesteps to use.
    """
    phi = np.zeros_like(v)
    generated_spikes = []

    for t in range(timesteps):
        phi += v
        phi_abs = np.abs(phi)
        while np.max(phi_abs) > 0.5:
            i = np.argmax(phi_abs)
            s = np.sign(phi[i])
            phi[i] -= s
            yield SignedSpike(source=i, sign=s, timestep=t)




def rectified_tensor_quantization(v, timesteps):
     """Quantize a vector in R^d into a sequence of positive events.
    c.f. Algorithm 2 in https://arxiv.org/pdf/1602.08323.pdf.

    The sequence will converge with rate 1/T.

    Args:
        v: numpy float tensor of arbitrary shape to be quantized.
        timesteps: total number of timesteps to use.
    """