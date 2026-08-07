import numpy as np
from src.markov_model.hmm import HiddenMarkovModel
import pytest

@pytest.fixture
def feature_matrix():
    return np.array([
        [0.08, 0.10, 0.12],
        [0.26, 0.09, 0.13],
        [0.40, 0.36, 0.50]
    ])

@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("expected_states", [2, 5, 10, 20])
def test_states_saved(expected_states):
    """
    Test that the number of states is correctly saved in the HiddenMarkovModel instance.
    """
    feature_matrix = np.zeros((expected_states, 2))
    hmm = HiddenMarkovModel(feature_matrix=feature_matrix, states=expected_states, covariance_type="full")

    assert hmm.states == expected_states


@pytest.mark.filterwarnings("ignore")
def test_mean_dim(feature_matrix):
    """
    Test that the means are correctly computed and have the expected dimensions.
    """
    states = 3
    hmm = HiddenMarkovModel(feature_matrix=feature_matrix, states=states, covariance_type="full")
    _, num_features = feature_matrix.shape

    assert (states, num_features) == hmm.means.shape


@pytest.mark.filterwarnings("ignore")
def test_covariances_full_dim(feature_matrix):
    """
    Test that the covariances are correctly computed and have the expected dimensions for full covariance type.
    """
    states = 3
    hmm = HiddenMarkovModel(feature_matrix=feature_matrix, states=states, covariance_type="full")
    _, num_features = feature_matrix.shape

    assert (states,num_features, num_features) == hmm.covariances.shape


@pytest.mark.filterwarnings("ignore")
def test_covariances_diag_dim(feature_matrix):
    """
    Test that the covariances are correctly computed and have the expected dimensions for diagonal covariance type.
    """
    states = 3
    hmm = HiddenMarkovModel(feature_matrix=feature_matrix, states=states, covariance_type="diag")
    _, num_features = feature_matrix.shape

    assert (states, num_features) == hmm.covariances.shape

@pytest.mark.filterwarnings("ignore")
def test_transition_matrix_sum_equals_one(feature_matrix):
    """
    Test that each row of the transition matrix sums to 1.
    """
    states = 3
    hmm = HiddenMarkovModel(feature_matrix=feature_matrix, states=states, covariance_type="full")

    assert np.allclose(hmm.transition_matrix.sum(axis=1),1.0)

@pytest.mark.filterwarnings("ignore")
def test_initial_probabilities_sum_equals_one(feature_matrix):
    """
    Test that the initial probabilities sum to 1.
    """
    states = 3
    hmm = HiddenMarkovModel(feature_matrix=feature_matrix, states=states, covariance_type="full")

    assert np.isclose(np.sum(hmm.init_prob), 1.0)

def test_invalid_covariance_type(feature_matrix):
    """
    Test that an invalid covariance type raises a ValueError.
    """
    with pytest.raises(ValueError):
        HiddenMarkovModel(
            feature_matrix,
            states=3,
            covariance_type="banana"
        )

def test_empty_feature_matrix():
    """
    Test that an empty feature matrix raises a ValueError.
    """

    feature_matrix = np.array([])
    with pytest.raises(ValueError):
        HiddenMarkovModel(
            feature_matrix,
            states=3,
            covariance_type="full"
        )


def test_states_equal_1(feature_matrix):
    """
    Test that specifying only one state raises a ValueError.
    """
    with pytest.raises(ValueError):
        HiddenMarkovModel(
            feature_matrix,
            states=1,
            covariance_type="full"
        )