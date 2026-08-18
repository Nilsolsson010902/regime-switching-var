import numpy as np
from src.markov_model.hmm import HiddenMarkovModel
import pytest

@pytest.fixture
def feature_matrix():
    rng = np.random.default_rng(42)

    state_0 = rng.normal(
        loc=[0.02, 0.05, 0.10],
        scale=[0.01, 0.02, 0.02],
        size=(50, 3)
    )

    state_1 = rng.normal(
        loc=[-0.03, -0.05, 0.30],
        scale=[0.02, 0.03, 0.04],
        size=(50, 3)
    )

    return np.vstack([state_0, state_1])

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
    Test that the smooth probabilities sum to 1 for each time step.
    """
    assert np.allclose(model_full.smooth_proba(X=X).sum(axis=1), 1.0)
    assert np.allclose(model_diag.smooth_proba(X=X).sum(axis=1), 1.0)

def test_probability_output_shapes(model_full, X):
    """
    Test that the output shapes of filter_proba and smooth_proba are correct.
    """
    assert model_full.filter_proba(X).shape == (len(X), 2)
    assert model_full.smooth_proba(X).shape == (len(X), 2)

def test_predict_output(model_full, X):
    """
    Test that output shapes of predict are correct. 
    """
    states = model_full.predict(X)

    assert states.shape == (len(X),)
    assert np.all(states >= 0)
    assert np.all(states < model_full.states)