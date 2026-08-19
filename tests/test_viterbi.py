import numpy as np

from src.markov_model.viterbi import viterbi


def test_viterbi_known_path():
    """
    Test the Viterbi algorithm with a known path and emission probabilities.
    """
    init_prob = np.array([0.9, 0.1])

    transition_matrix = np.array([
        [0.9, 0.1],
        [0.1, 0.9]
    ])

    emission_matrix = np.array([
        [0.95, 0.05],  # strongly favors state 0
        [0.90, 0.10],  # strongly favors state 0
        [0.10, 0.90],  # strongly favors state 1
        [0.05, 0.95]   # strongly favors state 1
    ])

    path = viterbi(
        init_prob=init_prob,
        transition_matrix=transition_matrix,
        log_emission_matrix=np.log(emission_matrix)
    )

    expected = np.array([0, 0, 1, 1])

    assert np.array_equal(path, expected)


def test_viterbi_output_shape():
    """
    Test that the output shape of the Viterbi algorithm is correct.
    """
    init_prob = np.array([0.5, 0.5])

    transition_matrix = np.array([
        [0.8, 0.2],
        [0.2, 0.8]
    ])

    log_emission_matrix = np.log(np.array([
        [0.7, 0.3],
        [0.6, 0.4],
        [0.2, 0.8]
    ]))

    path = viterbi(
        init_prob,
        transition_matrix,
        log_emission_matrix
    )

    assert path.shape == (3,)

def test_viterbi_states_are_valid():
    """
    Test that the output states of the Viterbi algorithm are valid.
    """
    init_prob = np.array([0.5, 0.5])

    transition_matrix = np.array([
        [0.8, 0.2],
        [0.2, 0.8]
    ])

    log_emission_matrix = np.log(np.array([
        [0.7, 0.3],
        [0.6, 0.4],
        [0.2, 0.8]
    ]))

    path = viterbi(
        init_prob,
        transition_matrix,
        log_emission_matrix
    )

    assert np.all(path >= 0)
    assert np.all(path < 2)

def test_viterbi_single_observation():
    """
    Test the Viterbi algorithm with a single observation.
    """
    init_prob = np.array([0.2, 0.8])

    transition_matrix = np.array([
        [0.9, 0.1],
        [0.1, 0.9]
    ])

    log_emission_matrix = np.log(np.array([
        [0.1, 0.9]
    ]))

    path = viterbi(
        init_prob,
        transition_matrix,
        log_emission_matrix
    )

    assert np.array_equal(path, np.array([1]))