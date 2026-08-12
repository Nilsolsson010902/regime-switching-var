import numpy as np
from src.markov_model.hmm import HiddenMarkovModel
import pytest

@pytest.fixture
def feature_matrix():
    return np.array([
        [-1.2, -0.8],
        [-1.0, -1.1],
        [-0.9, -0.7],
        [ 1.0,  0.8],
        [ 1.2,  1.1],
        [ 0.8,  1.0],
    ])

@pytest.mark.parametrize("covariance_type", ["full", "diag"])
def test_emission_matrix_shape(feature_matrix, covariance_type):
    """
    Test that the emission probability matrix has the correct shape and contains finite values.
    """
    hmm = HiddenMarkovModel(
        feature_matrix=feature_matrix,
        states=2,
        covariance_type=covariance_type
    )

    emissions = hmm.compute_emission_probability_matrix()

    assert emissions.shape == (len(feature_matrix), 2)

@pytest.mark.parametrize("covariance_type", ["full", "diag"])
def test_emissions_are_finite(feature_matrix, covariance_type):
    """
    Test that the emission probability matrix contains only finite values.
    """
    hmm = HiddenMarkovModel(
        feature_matrix=feature_matrix,
        states=2,
        covariance_type=covariance_type
    )

    emissions = hmm.compute_emission_probability_matrix()

    assert np.all(np.isfinite(emissions))


@pytest.mark.parametrize("covariance_type", ["full", "diag"])
def test_mean_has_higher_log_emission_than_far_observation(feature_matrix, covariance_type):
    """
    Test that the log emission probability of an observation at the mean is higher than that of a far observation.
    """
    hmm = HiddenMarkovModel(
        feature_matrix=feature_matrix,
        states=2,
        covariance_type=covariance_type
    )

    state = 0

    observation_at_mean = hmm.means[state]
    far_observation = hmm.means[state] + 10.0

    log_prob_mean = hmm.log_emission_probability(
        observation_at_mean,
        state
    )

    log_prob_far = hmm.log_emission_probability(
        far_observation,
        state
    )

    assert log_prob_mean > log_prob_far


