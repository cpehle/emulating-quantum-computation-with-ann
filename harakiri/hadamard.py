from . import unitary_2d
from . import quantum

def train_hadamard():
    model,history = unitary_2d.train_model(
        unitary_transform=quantum.hadamard,
        epochs=3000, 
        num_samples=10000, 
        batch_size=10000, 
        model=unitary_2d.build_linear_model([20,30,4], optimizer='adam'))
    return model,history
