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
        self.mean = None
        self.initialize_parameters()


    def initialize_parameters(self) -> None:
        kmeans = KMeans(n_clusters=self.states)
        kmeans.fit(self.feature_matrix)
        self.means = kmeans.cluster_centers_
        labels = kmeans.labels_
   
        self.init_prob = np.array([1/self.states], ndim=self.states)
        for i in range(self.states):
        
            transition_i = []
            self.covariances = np.append(self.covariances, (kmeans.cluster_centers_[i] -self.means[i])**2/len(kmeans.cluster_centers_[i] -1))


