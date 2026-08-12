import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class HiddenMarkovModel:
    def __init__(self, feature_matrix: np.ndarray, states: int, covariance_type: str = "full"):
        self.covariance_type = covariance_type.lower()
        self.feature_matrix = np.array(feature_matrix)
        self.states = states 
        self.transition_matrix = None
        self.init_prob = None
        self.covariances = None
        self.means = None
        self.initialize_parameters()


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
            covariance_matrices.append(covariance)

        if self.covariance_type == "diag":
            self.covariances= np.array([np.diag(covariance_matrix) for covariance_matrix in covariance_matrices])
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


    def compute_emission_probability_matrix(self) -> np.ndarray:
        """
        Compute the emission probability matrix for all observations and states.

        Returns
        -------
        np.ndarray
            The emission probability matrix.
        """
        probability_matrix = []
        for feature in self.feature_matrix:
            row = []
            for state in range(self.states):
                
                state_emission_prob = self.log_emission_probability(observation=feature, state=state)

                row.append(state_emission_prob)

            probability_matrix.append(row)
        return np.array(probability_matrix)
            
    #def fit(self):
