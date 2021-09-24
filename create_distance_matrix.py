
import numpy as np
from numba import jit


@jit(nopython=True)
def cosdist(A):
    '''
    calculates the cosince distance between all rows in an array.
        returns the resulting distance matrix
    '''
    
    D = np.zeros((len(A),len(A)))
    
    for i in range(1,len(A)):
        for j in range(i):
            D[i,j] = np.dot(A[i], A[j].T)/(np.linalg.norm(A[i]) * np.linalg.norm(A[j]))
            D[j,i] = D[i,j]
    
    return D

@jit(nopython=True)
def l2dist(A):
    '''
    calculates the euclidian (l2) distance between all rows in an array.
        returns the resulting distance matrix
    '''
    
    D = np.zeros((len(A),len(A)))
    
    for i in range(1,len(A)):
        for j in range(i):
            D[i,j] = np.sqrt(np.sum(np.square( A[i]-A[j] )))
            D[j,i] = D[i,j]
    
    return D



class matobj():
    
    def __init__(self,  N,
                        H,
                        mean=None,
                        initiate=True):
        
        self.N      = N
        self.H      = H
        self.mean   = mean
        
        if initiate:
            self.initiate()    
    
    def initiate(self):     
        # initiate to random state
        self.matrix = np.random.randn(self.N,self.N) + self.mean
        
        # make symetric by averaging
        self.matrix = ( self.matrix + self.matrix.T ) / 2
        
        # set diagonal to zero
        np.fill_diagonal(self.matrix, 0)
    
    def from_point_cloud(self, points, npert):
        '''
        2D points in space is perturbed by adding a small quantity to the first instance (x)
        '''
        
        # initiate array
        A = np.zeros((len(points)*npert,2))
        
        # for each point in points
        for ip,point in enumerate(points):
            
            # perturb point n times
            xy = point[0]
            for i in range(npert):
                point[0] = xy + 0.3*i
                point[1] = xy + 0.3*i
                
                # add to array
                A[ip*npert+i] = point
        
        # calc distance matrix of array
        self.matrix = l2dist(A)
        self.matrix = self.matrix / np.max(self.matrix) * (self.H-1)
        
    def random_from_XYZ_plain(self, dim=3):
        
        # create random x, y and z vectors of len N
        # calc distance for each pair
        
        self.matrix = cosdist( np.random.rand(self.N, dim) ) * self.H
    
    def multimodal(self,    ngroups=2,
                            max_between_percentage=10,
                            max_within_percentage=80,
                            min_within_percentage=50):
        
        # check so that at least one pattern for each group can be created
        if self.N < ngroups:
            print("Argument ERROR: N is smaller than ngroups")
            raise
        
        dx = int(np.floor(self.N/ngroups))
        
        # transform percentage overlap to distance limits
        bm   = self.H - int(np.floor(self.H * max_between_percentage / 100))
        wmin = self.H - int(np.floor(self.H * max_within_percentage  / 100))
        wmax = self.H - int(np.floor(self.H * min_within_percentage  / 100)) + 1
        
        # initiate mother frame
        self.matrix = np.random.randint(bm, high=self.H+1, size=(self.N,self.N))
        
        # initiate individual groups and add to frame
        for i in range(ngroups):
            self.matrix[i*dx:(i+1)*dx,i*dx:(i+1)*dx] = \
                np.random.randint(  wmin,
                                    wmax, 
                                    size=(dx,dx))
        
        # set diagonal to 0
        np.fill_diagonal(self.matrix, 0)
        
        # make symetrical
        self.matrix = ( self.matrix + self.matrix.T ) / 2
        
    
        
        
        
    
    
    

