
import numpy as np


class matobj():
    
    def __init__(self,  N=10,
                        mean=None):
        
        # initiate to random state
        self.matrix = np.random.randn(N,N) + mean
        
        # make symetric by averaging
        self.matrix = ( self.matrix + self.matrix.T ) / 2
        
        # set diagnol to zero
        np.fill_diagonal(self.matrix, 0)
    
    def from_point_cloud(self):
        pass
    
        
        
        
    
    
    

