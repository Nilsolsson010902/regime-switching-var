import numpy as np
from src.markov_model.hmm import HiddenMarkovModel
import pytest

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("expected_states", [2, 5, 10, 10])
def test_states_saved(expected_states):
    """
    Test that the number of states is correctly saved in the HiddenMarkovModel instance.
    """
    feature_matrix = np.zeros((expected_states, 2))
    hmm = HiddenMarkovModel(feature_matrix=feature_matrix, states=expected_states, covariance_type="full")

    assert hmm.states == expected_states


@pytest.mark.filterwarnings("ignore")
def test_mean_dim():
    """
    Test that the means are correctly computed and have the expected dimensions.
    """
    feature_matrix =  np.array([
            [0.08, 0.10, 0.12],
            [0.26, 0.09, 0.13],
            [0.40, 0.36, 0.5] 
            ])
    states = 3
    hmm = HiddenMarkovModel(feature_matrix=feature_matrix, states=states, covariance_type="full")
    observation_days, num_features = feature_matrix.shape

    assert (states, num_features) == hmm.means.shape


@pytest.mark.filterwarnings("ignore")
def test_covariances_full_dim():
    """
    Test that the covariances are correctly computed and have the expected dimensions for full covariance type.
    """
    feature_matrix =  np.array([
                [0.08, 0.10, 0.12],
                [0.26, 0.09, 0.13],
                [0.40, 0.36, 0.5] 
                ])
    states = 3
    hmm = HiddenMarkovModel(feature_matrix=feature_matrix, states=states, covariance_type="full")
    observation_days, num_features = feature_matrix.shape

    assert (states,num_features, num_features) == hmm.covariances.shape


@pytest.mark.filterwarnings("ignore")
def test_covariances_full_dim():
    """
    Test that the covariances are correctly computed and have the expected dimensions for diagonal covariance type.
    """
    feature_matrix =  np.array([
                [0.08, 0.10, 0.12],
                [0.26, 0.09, 0.13],
                [0.40, 0.36, 0.5] 
                ])
    states = 3
    hmm = HiddenMarkovModel(feature_matrix=feature_matrix, states=states, covariance_type="diag")
    observation_days, num_features = feature_matrix.shape

    assert (states, num_features) == hmm.covariances.shape

@pytest.mark.filterwarnings("ignore")
def test_transition_matrix_sum_equals_one():
    """
    Test that each row of the transition matrix sums to 1."""
    feature_matrix =  np.array([
                [0.08, 0.10, 0.12],
                [0.26, 0.09, 0.13],
                [0.40, 0.36, 0.5] 
                ])
    states = 3
    hmm = HiddenMarkovModel(feature_matrix=feature_matrix, states=states, covariance_type="full")

    for row in hmm.transition_matrix:
        assert np.sum(row) == 1

@pytest.mark.filterwarnings("ignore")
def test_initial_probabilities_sum_equals_one():
    """
    Test that the initial probabilities sum to 1.
    """
    feature_matrix =  np.array([
                [0.08, 0.10, 0.12],
                [0.26, 0.09, 0.13],
                [0.40, 0.36, 0.5] 
                ])
    states = 3
    hmm = HiddenMarkovModel(feature_matrix=feature_matrix, states=states, covariance_type="full")

    assert np.sum(hmm.init_prob) == 1