import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class HiddenMarkovModel:
    def __init__(self, feature_matrix, states: int):
        self.feature_matrix = feature_matrix
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

        if self.feature_matrix is None or self.feature_matrix.size == 0:
            raise ValueError("Feature matrix is empty. Cannot initialize HMM parameters.")

        if self.states <= 0:
            raise ValueError("Number of states must be a positive integer.")
        
        kmeans = KMeans(n_clusters=self.states, random_state=42, n_init=10)
        kmeans.fit(self.feature_matrix)
        self.means = kmeans.cluster_centers_
        labels = kmeans.labels_
        self.init_prob = np.array([1/self.states]*self.states)

        #Initial stay probability high as market regimes often dont switch
        stay_probability = 0.9
        switch_probability = (1-stay_probability)/(self.states-1)
        self.transition_matrix = np.array([[switch_probability]*self.states]* self.states)
        np.fill_diagonal(self.transition_matrix, stay_probability)

        covariance_matrices = []

        for i in range(self.states):
            state_data = self.feature_matrix[labels == i]

            covariance = np.cov(state_data, rowvar=False)
            covariance_matrices.append(covariance)

        self.covariances = np.array(covariance_matrices)


