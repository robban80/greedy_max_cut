

''' implement a greedy max cut sorting of distances into Hc/Mc patterns.
    Robert Lindroos 2020-12-08:->
'''

from time import time
import numpy as np
import matplotlib.pyplot as plt

import pickle
import json



features = ['Pleasantness', 'Intensity', 'Familiarity', 'Edability']


def run():
    
    for i in range(2):
        for fp in range(1,38):
            main(i,fp)


def main(i,fp):    
    # based on a distance matrix (D):
    # pre1. (find pattern with largest averge overlap, p0)
    # pre2. (sort patterns based on higest overlap with p0)
    # 1. initialize. set p0 = 0 for all Hc. 
    #                Random values for other patterns gives best result, all zero also work.
    # 2. fill pattern matrix column-vice based on cost and overlap
    # 3. optimze: repeatedly flip trough the matrix (using greedy strategy)
    # 4. TODO: (randomly permutate Hc and mc's)
    
    # create pattern
    #Dn = D_from_xy_plane()
    #check_trinequality(D)
    # pre1 and pre2
    #Dn = resort_distances(D)
    #check_trinequality(Dn)
    
    non_standard = 1
    perc = features[i]
    path = '../../../../Box/Odor_similarity_data/Individual_distance_matrices/'
    if not fp: fp = 6
    
    # load distance matrix
    if non_standard:
        fname = '{}D_fp{}_{}.json'.format(path,fp,perc)
        save_fname = 'Patterns/All_participants/pattern_fp{}_{}'.format(fp,perc)
        with open(fname, 'r') as f:
            Dn = np.array(json.load(f))
    else:
        save_fname = 0
        with open('D.json', 'r') as f:
            Dn = np.array(json.load(f))
    
    # print(Dn)
    #check_trinequality(Dn)
        
    # 1,2
    P = set_patterns_Q(Dn, Nr=5)
    
    #with open('Pinit.pkl', 'wb') as outfile:
    #    pickle.dump({'D':D, 'Dn':Dn, 'P':P}, outfile)
    
    # plot patterns
    #plot_(Dn,P)
    
    # 3
    # optimize
    upd_pattern_Q(P,Dn, saveP=1, save_fname=save_fname)
    
    #plt.show()
    plt.close('all')

def upd_pattern(P,Dn, N=10, n=12, saveP=0, perc=0):    
    ''' 
    iteratively "flip" values of idividual hypercolumns 
    easily gets stuck in local minima. random flipping is added to minimize this risk.
    '''
    
    error = np.zeros(N+1)
    error[0] = hamming_distance(P,Dn)
    best = 100
    for i in range(N):
        
        P           = set_patterns(Dn, P=P)
        error[i+1]  = hamming_distance(P,Dn)
        
        print(i, error[i+1])
        
        if error[i+1] == error[i]: 
            # add random perturbations
            x = np.random.randint(P.shape[0], size=n)
            y = np.random.randint(P.shape[1], size=n)
            print(x)
            for j in range(len(x)):
                P[x[j],y[j]] = np.random.randint(10)
            
            if n >= 2: n = int(n/2)
        
        if error[i+1] < best:
            Pbest = P.copy()
            best = error[i+1]
            
    # plot patterns
    plot_(Dn,Pbest, error=error, saveFig=0)
    
    if saveP:
        if perc:
            with open('patterns_{}.json'.format(perc), 'w') as f:
                json.dump(Pbest.astype(int).tolist(), f)
        else:
            with open('patterns.json', 'w') as f:
                json.dump(Pbest.astype(int).tolist(), f)
    
def upd_pattern_Q(P,Dn, saveP=0, save_fname=0):    
    ''' 
    iteratively "flip" values of idividual hypercolumns 
    easily gets stuck in local minima. random flipping is added to minimize this risk.
    '''
    
    Nr = [5,5,4,4,3,3,2,2,1,1,0,0,0,0] #  [0,0,0,0,0] # 
    Nr = [5,5,4,4,3,3,3,2,2,2,1,1,1,0,0,0,0,0]
    N  = len(Nr)
    error = np.zeros(N+1)
    error[0] = hamming_distance(P,Dn)
    best = 100
    for i,n in enumerate(Nr):
        
        P           = set_patterns_Q(Dn, P=P, Nr=n)
        error[i+1]  = hamming_distance(P,Dn)
        
        print(i, error[i+1])
        
        if error[i+1] == error[i]: 
            # add random perturbations
            x = np.random.randint(P.shape[0], size=n)
            y = np.random.randint(P.shape[1], size=n)
            
            for j in range(len(x)):
                P[x[j],y[j]] = np.random.randint(10)
            
            if n >= 2: n = int(n/2)
        
        elif error[i+1] < best:
            Pbest = P.copy()
            best = error[i+1]
        
        
        # randomly shuffle columns
        np.random.shuffle(P.T)
    
    if saveP:
        emax = 1    #0.5
        if best < emax:
            # save patterns...
            if save_fname:
                with open('{}.json'.format(save_fname), 'w') as f:
                    json.dump(Pbest.astype(int).tolist(), f)
            else:
                with open('patterns.json', 'w') as f:
                    json.dump(Pbest.astype(int).tolist(), f) 
            # and plot
            plot_(Dn,Pbest, error=error, saveFig=1, save_fname=save_fname)
            
        else:
            print('--- error={:.2f} > {} --- reinitiating -----'.format(best, emax))
            P = set_patterns_Q(Dn, Nr=5)
            upd_pattern_Q(P,Dn,saveP=1,save_fname=save_fname)
            
            


def plot_(D,P, error=[], saveFig=0, save_fname='patfig'):
    
    d,e = hamming_distance(P,D, return_distances=1 )
    print()
    print(e)
    
    fig,ax = plt.subplots(2,3, figsize=(10,8))
    
    ax[0,0].imshow(D, interpolation='none', aspect='equal', vmin=0, vmax=10, cmap='Greys_r')
    ax[0,0].set_title('Original distances')
    plot_patterns(P, ax=ax[0,1])
    ax[0,1].set_title('Resulting patterns')
    ax[0,1].set_ylim([-1,10])
    ax[0,2].imshow(d, interpolation='none',aspect='equal', vmin=0, vmax=10, cmap='Greys_r')
    ax[0,2].set_title('From patterns. HE = {}'.format(np.around(e,2)))
    
    ax[1,2].imshow(np.abs(d-D), interpolation='none',aspect='equal', vmin=0, vmax=10, cmap='Greys')
    ax[1,2].set_ylabel('diff |Org-dPatterns|')
    
    ax[1,1].axis('off')
    
    if len(error):
        ax[1,0].plot(error)
        ax[1,0].set_ylabel('Hamming error')
        #ax[1,0].set_ylim([0,1])
    
    for i in [0,2]: 
        ax[0,i].set_yticks([0,5,10,15])
        ax[0,i].set_ylim([15.5,-0.5])
    ax[1,2].set_yticks([0,5,10,15])
    ax[1,2].set_ylim([15.5,-0.5])
    
    if saveFig:
        if save_fname:
            plt.savefig(save_fname, dpi=300)
        else:
            plt.savefig('../../../../Box/Latex/Figures/setPattern_GMC_May21', dpi=300)
    


def set_patterns(D, P=[], z=0):
    # fill pattern matrix column-vice (one Hc at a time)
    #   **** This is the core of the algorithm ****
    
    if not len(P):
        # initialize
        if z:
            P = np.zeros((len(D),10))
        else:
            P = np.random.uniform(10, size=(len(D),10))
            P[0] = np.zeros(10)
    
    # set patterns using greedy algorithm.
    for j in range(10):
        for i in range(1,16):
            #       calc cost of setting P[i,j] to 0-9 
            C = np.zeros(10)
            for m in range(10):
                P[i,j] = m
                C[m] = hamming_distance(P,D)
            P[i,j] = np.argsort(C)[0]
    
    return P


def set_patterns_Q(D, P=[], z=0, Nr=0):
    # fill pattern matrix column-vice (one Hc at a time)
    #   **** This is the core of the algorithm ****
    # -> adds a portion af randomly set columns in each hypercolumn
    
    if not len(P):
        # initialize
        if z:
            P = np.zeros((len(D),10))
        else:
            P = np.random.uniform(10, size=(len(D),10))
            P[0] = np.zeros(10)
    
    # set patterns using greedy algorithm.
    for j in range(10):
        for i in range(1,16):
            #       calc cost of setting P[i,j] to 0-9 
            C = np.zeros(10)
            for m in range(10):
                P[i,j] = m
                C[m] = hamming_distance(P,D)
            
            # select the value with the shortest distance
            P[i,j] = np.argsort(C)[0]
    
    if Nr:
        # randomly perturb patterns
        
        # create random matrix with same shape
        R = np.random.randint(10, size=P.shape)
        
        # create matrix with zeros and ones used to select subportions of P and R
        r01 = np.ones(P.shape)
        r01[:Nr,:] = 0
        r01 = np.random.permutation(r01.flatten()).reshape(P.shape)
        
        P = P*r01 + R*(1-r01)
    
    return P


def hamming_distance(a,D, M=10, return_distances=0):
    
    '''
    a holds the index of the minicolumns for each hypercolumn and pattern.
        if two patterns hold the same value for a single Hc, they have an overlap of 1.
        -e.g. [1,2,3] and [2,2,3] would have an overlap of 2
        
        The distance is then the number of minicolumn (M) - overlap, e.g. 10 - 2 = 8   
    '''
    
    # TODO perhaps there is some more efficient algorithm for this?
    
    npat    = a.shape[0]
    d       = np.zeros((npat,npat))
    
    for i in range(npat):
        for j in range(i+1):
            
            d[i,j] = M - np.sum([1 for jj in range(a.shape[1]) if a[i,jj] == a[j,jj]])
            d[j,i] = d[i,j]
    
    if return_distances:
        return d, np.mean(np.abs(d-D))
    else:       
        return np.mean(np.abs(d-D))
          

def D_from_xy_plane():
    
    # create distance matrix satisfying triangle inequality
    
    X = np.random.randint(8, size=16)
    Y = np.random.randint(8, size=16)
    
    d = np.zeros((16,16))
    
    for i in range(1,16):
        for j in range(i):
            d[i,j] = np.sqrt( np.square(X[j]-X[i]) + np.square(Y[j]-Y[i]) )
            d[j,i] = d[i,j]
    
    # this could potentially destroy the inequality? Rounding does destroy
    d = np.ceil(d)  
    
    '''
    x = list(X[0:3]) + [X[0]]
    y = list(Y[0:3]) + [Y[0]]
    plt.plot(x, y, c='r', lw=2)
    plt.plot(X,Y, 'o', c='grey', mec='k', ms=10, alpha=0.8)
    plt.savefig('../../../../Box/Latex/Figures/setPattern_randomPointsXY', dpi=300)
    plt.show()'''
    
    return d       


def resort_distances(D):
    
    # find pattern largest average overlap, p0
    mean = np.mean(D, axis=0)   # axis could be 1 as well, since symetric...
    p0 = np.argmin(mean)     
    
    # sort patter based on similarity with p0
    si = np.argsort(D[p0])
    
    # re-map D based on p0-pN (N=15)
    Dn = np.zeros(D.shape)
    for i in range(len(D)):
        for j in range(len(D)):
            Dn[si[i],si[j]] = D[i,j]
    
    return(Dn)
            


def check_trinequality(D):
    
    N = len(D)
    for i in range(N):
        for j in range(i):
            assert all(D[i,j] <= D[i,:] + D[:,j])
    


def plot_patterns(h, ax=None):
    
    N = len(h)
    Cgrad = [[0,n/(N-1),n/(N-1)] for n in range(N)]
    
    if not ax:
        fig,ax = plt.subplots(1,1)
    for i in range(h.shape[0]):
        ax.plot(range(10), h[i], '.-', 
                        ms=16, mec='k', lw=1, markerfacecolor='w',
                        label='p{}'.format(i), c=Cgrad[i])
    
    ax.set_xlabel('hypercolumns')
    ax.set_ylabel('microcolumns')
    

run()
