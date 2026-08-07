import numpy as np
from src.markov_model.forward import forward_filtering

def test_forward_filtering():
    """
    
    Test the forward filtering algorithm with a simple example.

    """
    initial_probabilities = np.array([0.5, 0.5])

    transition_matrix = np.array([
        [0.95, 0.05],
        [0.20, 0.80]
    ])

    emission_matrix = np.array([
        [0.90, 0.10],
        [0.20, 0.80],
        [0.10, 0.90]
    ])

    log_emissions = np.log(emission_matrix)
    log_alpha = forward_filtering(initial_probabilities, transition_matrix, log_emissions)
    alpha = np.exp(log_alpha)

    expected = np.array([
        [0.9000, 0.1000],
        [0.6364, 0.3636],
        [0.1889, 0.8111]
    ])

    assert np.allclose(alpha, expected, atol=1e-3)