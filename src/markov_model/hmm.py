import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from .forward import forward_filtering
from .backward import backward_filtering
from .baum_welch import baum_welch_step
from .viterbi import viterbi
from scipy.special import logsumexp

class HiddenMarkovModel:
    def __init__(self, feature_matrix: np.ndarray, states: int, covariance_type: str = "full"):
        self.covariance_type = covariance_type.lower()
        self.feature_matrix = np.array(feature_matrix)
        self.states = states 
        self.transition_matrix = None
        self.init_prob = None
        self.covariances = None
        self.means = None
        self.log_likelihood_history = None
        self.initialize_parameters()
        self.is_fitted = False


    def initialize_parameters(self) -> None:
        """
        Initialize the parameters of the Hidden Markov Model using KMeans clustering.
        
        This method sets the initial state probabilities, transition matrix, means, and covariance matrices
        based on the provided feature matrix and the number of states.
        """
        if self.states <= 1:
                    raise ValueError("States must be an integer larger than 1")
                
        if self.covariance_type not in ["full", "diag"]:
            raise ValueError("Covariance type must be either 'full' or 'diag'.")
        
        if self.feature_matrix is None or self.feature_matrix.size == 0:
            raise ValueError("Feature matrix is empty. Cannot initialize HMM parameters.")
        
        kmeans = KMeans(n_clusters=self.states, random_state=42, n_init=10)
        kmeans.fit(self.feature_matrix)
        self.means = kmeans.cluster_centers_
        labels = kmeans.labels_
        self.init_prob = np.full(self.states, 1.0 / self.states)  # Uniform initial probabilities

        stay_probability = 0.9  #Initial stay probability high as market regimes often dont switch
        switch_probability = (1-stay_probability)/(self.states-1)
        self.transition_matrix = np.array([[switch_probability]*self.states]* self.states)
        np.fill_diagonal(self.transition_matrix, stay_probability)

        covariance_matrices = []

        for i in range(self.states):
            state_data = self.feature_matrix[labels == i]

            covariance = np.cov(state_data, rowvar=False)
            #regularization
            covariance += 1e-10 * np.eye(covariance.shape[0])
            covariance_matrices.append(covariance)

        if self.covariance_type == "diag":
            self.covariances= np.array([
                np.maximum(np.diag(covariance_matrix), 1e-10)
                for covariance_matrix in covariance_matrices
                ])
        else: 
            self.covariances = np.array(covariance_matrices)

    def log_emission_probability(self, observation: np.ndarray, state: int) -> float:
        """
        Compute the log emission probability of an observation given a state.

        Parameters
        ----------
        observation : np.ndarray
            The observed feature vector.
        state : int
            A given state.

        Returns
        -------
        float
            The log emission probability.
        """
        d = len(observation) 
        dim_const = d * np.log(np.pi * 2)
        if self.covariance_type == "full":
            
            log_determinant = np.log(np.linalg.det(self.covariances[state]))
            mahalanobis = np.subtract(observation, self.means[state]).T @ np.linalg.inv(self.covariances[state]) @ np.subtract(observation, self.means[state])
            
        else: 
            log_determinant = np.sum(np.log(self.covariances[state]))
            mahalanobis = np.sum(np.subtract(observation, self.means[state])**2 /self.covariances[state] )
        
        return -0.5*(dim_const + log_determinant+ mahalanobis)


    def compute_emission_probability_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the emission probability matrix for all observations and states.

        Returns
        -------
        np.ndarray
            The emission probability matrix.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional array.")

        if X.shape[1] != self.means.shape[1]:
            raise ValueError( f"Expected {self.means.shape[1]} features, got {X.shape[1]}.")

        if not np.all(np.isfinite(X)):
            raise ValueError("X contains NaN or infinite values.")
        
        probability_matrix = []
        for feature in X:
            row = []
            for state in range(self.states):
                
                state_emission_prob = self.log_emission_probability(observation=feature, state=state)

                row.append(state_emission_prob)

            probability_matrix.append(row)
        return np.array(probability_matrix)
    
            
    def fit(self, max_itr = 100, tol = 1e-5):
        """
        Fit the Hidden Markov Model to the feature matrix using the Baum-Welch algorithm.

        Parameters
        ----------
        max_itr : int, optional
            Maximum number of iterations for the Baum-Welch algorithm. Default is 100.
        tol : float, optional
            Tolerance for convergence. Default is 1e-5.

        Returns
        -------
        HiddenMarkovModel
            The fitted Hidden Markov Model instance.
        """
        old_ll = (-np.inf)
        self.log_likelihood_history = []

        for itr in range(max_itr):
            log_emissions = self.compute_emission_probability_matrix(X = self.feature_matrix)
            new_ll = forward_filtering(initial_probabilities=self.init_prob, 
                                    transition_matrix=self.transition_matrix, 
                                    log_emission_matrix=log_emissions).log_likelihood
            
            self.log_likelihood_history.append(new_ll)
            #if conversion then break
            if(np.abs(old_ll - new_ll) < tol):
                break

            bw_output = baum_welch_step(initial_probabilities=self.init_prob, transition_matrix=self.transition_matrix, log_emission_matrix=log_emissions, feature_matrix=self.feature_matrix, covariance_type=self.covariance_type)

            #update hmm properties
            self.init_prob = bw_output.initial_probabilities
            self.transition_matrix = bw_output.transition_matrix
            self.means = bw_output.means
            self.covariances = bw_output.covariances
            old_ll = new_ll

        self.is_fitted = True
        return self

    def filter_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the filtered state probabilities for the given observations.
        """
        if not self.is_fitted:
                    raise ValueError("Modell not fitted yet")
        
        log_emissions = self.compute_emission_probability_matrix(X = X)
        return np.exp(forward_filtering(initial_probabilities=self.init_prob, 
                                transition_matrix=self.transition_matrix, 
                                log_emission_matrix=log_emissions).log_alpha)


    def smooth_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the smoothed state probabilities for the given observations.
        """
        if not self.is_fitted:
            raise ValueError("Modell not fitted yet")
        
        log_emissions = self.compute_emission_probability_matrix(X = X)
        log_alpha = forward_filtering(initial_probabilities=self.init_prob, 
                                        transition_matrix=self.transition_matrix, 
                                        log_emission_matrix=log_emissions).log_alpha

        log_beta = backward_filtering(transition_matrix=self.transition_matrix,
                                        log_emission_matrix=log_emissions)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)

        return np.exp(log_gamma)
    

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the most likely hidden state at each observation
        using smoothed state probabilities.
        """
        probs = self.smooth_proba(X=X)
        return np.argmax(probs, axis=1)
    

    def decode_viterbi(self, X: np.ndarray):
        """
        Decode the most likely hidden-state sequence using the Viterbi algorithm.
        """
        if not self.is_fitted:
                    raise ValueError("Modell not fitted yet")

        log_emission = self.compute_emission_probability_matrix(X=X)
        return viterbi(init_prob=self.init_prob, transition_matrix=self.transition_matrix, log_emission_matrix=log_emission)