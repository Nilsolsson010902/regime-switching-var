import numpy as np
from src.markov_model.hmm import HiddenMarkovModel
import pytest

@pytest.fixture
def feature_matrix():
    return np.array([
        [0.08, 0.10, 0.12],
        [0.26, 0.09, 0.13],
        [0.40, 0.36, 0.50],
        [0.10, 0.08, 0.11],

        [0.38, 0.35, 0.48],
        [0.40, 0.36, 0.50],
        [0.42, 0.34, 0.52],
        [0.39, 0.38, 0.47],
    ])

@pytest.fixture
def X():
    return np.array([
            [0.09, 0.12, 0.15],
            [0.35, 0.14, 0.07],
            [0.29, 0.36, 0.42]
        ])

@pytest.fixture
def model_full(feature_matrix):
    return HiddenMarkovModel(feature_matrix=feature_matrix, states=2, covariance_type="full").fit()

@pytest.fixture
def model_diag(feature_matrix):
    return HiddenMarkovModel(feature_matrix=feature_matrix, states=2, covariance_type="diag").fit()

def test_filter_probabilities_sum_to_one(model_diag, model_full, X):
    """
    Test that the filter probabilities sum to 1 for each time step.
    """
    assert np.allclose(model_full.filter_proba(X=X).sum(axis=1), 1.0)
    assert np.allclose(model_diag.filter_proba(X=X).sum(axis=1), 1.0)


def test_filter_probabilities_sum_to_one(model_diag, model_full, X):
    """
    Test that the smooth probabilities sum to 1 for each time step."""
    assert np.allclose(model_full.smooth_proba(X=X).sum(axis=1), 1.0)
    assert np.allclose(model_diag.smooth_proba(X=X).sum(axis=1), 1.0)

def test_probability_output_shapes(model_full, X):
    """
    Test that the output shapes of filter_proba and smooth_proba are correct."""
    assert model_full.filter_proba(X).shape == (len(X), 2)
    assert model_full.smooth_proba(X).shape == (len(X), 2)