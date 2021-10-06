

''' implement a greedy max cut sorting of distances into Hc/Mc patterns.
    Robert Lindroos 2020-12-08:->
'''

from time import time
import numpy as np
import matplotlib.pyplot as plt

import pickle, json, sys

import create_distance_matrix as dmat

from numba import jit



features = ['Pleasantness', 'Intensity', 'Familiarity', 'Edability']


def run_all_individual(emax=0.5):
    
    for i in range(2):
        for fp in range(1,38):
            main(non_standard=1, i=i, fp=fp, emax=emax)


def main(H=10, M=10, npat=20, mean_overlap=7, non_standard=4, i=None, fp=None, emax=0.5):    
    # based on a distance matrix (D):
    # 1. initialize. set p0 = 0 for all Hc. 
    #        Random values for other patterns gives best result, all zero also work.
    # 2. fill pattern matrix column-vice based on cost and overlap
    # 3. optimze: repeatedly flip trough the matrix (using greedy strategy)
    
    # create pattern
    #Dn = D_from_xy_plane()
    #check_trinequality(D)
    # pre1 and pre2
    #Dn = resort_distances(D)
    #check_trinequality(Dn)
    
    if non_standard==1:
        perc = features[i]
        path = '../../Box/Odor_similarity_data/Individual_distance_matrices/'
        path = 'InputDistances/'
        if not fp: fp = 6
        fname = '{}D_fp{}_{}.json'.format(path,fp,perc)
        save_fname = 'Patterns/All_participants/pattern_fp{}_{}'.format(fp,perc)
        with open(fname, 'r') as f:
            Dn = np.array(json.load(f))
    elif non_standard==2:
        Dn = dmat.matobj(npat, H, mean=mean_overlap).matrix  
    elif non_standard==3:
        obj = dmat.matobj(npat, H, mean=mean_overlap, initiate=False)
        obj.random_from_XYZ_plain(H=H)
        Dn = obj.matrix
    elif non_standard==4:
        obj = dmat.matobj(npat, H, mean=mean_overlap, initiate=False)
        ngroups = 10
        sep_b_g = 10
        sep_w_g = 2
        #obj.from_point_cloud(npat, ngroups, sep_b_g, sep_w_g, init_max_percentage=10, dist='l2')
        obj.from_point_cloud(npat, ngroups, sep_b_g, sep_w_g)
        Dn = obj.matrix   
        save_fname = 'OutputPatterns/multi_D{}-v2_npat{}_ngroups{}_emax{}'.format(
                        non_standard, 
                        npat,
                        ngroups, 
                        emax).replace('.','')
    elif non_standard==5:
        obj = dmat.matobj(npat, H, initiate=False)
        ng = 3
        obj.multimodal( ngroups=ng,
                        max_between_percentage=10,
                        max_within_percentage=60,
                        min_within_percentage=40)
        Dn = obj.matrix  
        save_fname = 'OutputPatterns/multi_D{}_npat{}_ngroups{}_emax{}'.format(
                        non_standard, 
                        npat,
                        ng, 
                        emax).replace('.','')
    else:
        with open('D_average.json', 'r') as f:
            Dn = np.array(json.load(f))
    
    print(Dn)
        
    npat = len(Dn)
    
    if non_standard not in [1,4,5]: 
        save_fname = 'OutputPatterns/multi_D{}_npat{}_emax{}'.format(non_standard, 
                                                                        npat, 
                                                                        emax).replace('.','')
    
    #check_trinequality(Dn)
        
    # 1,2
    P = set_patterns(Dn, H, M, Nr=5)
    
    #with open('Pinit.pkl', 'wb') as outfile:
    #    pickle.dump({'D':D, 'Dn':Dn, 'P':P}, outfile)
    
    # 3
    # optimize
    upd_pattern(P,Dn, H, M, saveP=1, save_fname=save_fname, emax=emax)
    
    #plt.show()
    if non_standard==1:
        plt.close('all')
    else:
        plt.show()

    
def upd_pattern(P,Dn, H, M, N=False, saveP=False, save_fname=False, emax=0.5):    
    ''' 
    iteratively "flip" values of idividual columns to minimize the global error
    
    easily gets stuck in local minima.
        random flipping is added to get out off such states
    
    emax sets the maximal mean error allowed. 
        A small value make it harder for the algorithm to compleate (it might even be impossible)
    '''
    
    if not N:
        Nr = [5,5,4,4,3,3,3,2,2,2,1,1,1]  # [3,2,1,0,0,0] # 
        N  = len(Nr)
        randomize = True
    else:
        randomize = False
        
    error       = np.zeros(N+1)
    error[0]    = hamming_distance(P,Dn,H)
    best        = 100      #initialize to high value (arbitrary)
    
    for i,n in enumerate(Nr):
        
        # update patterns
        if randomize:
            # set patterns allow shuffeling...
            P       = set_patterns(Dn, H, M, P=P, Nr=n)
        else:
            #... while the greedy_update only, does not
            P       = greedy_update(D, P, H, M)
        
        error[i+1]  = hamming_distance(P,Dn,H)
        
        print(i, error[i+1])
        
        # if stuck in local minima
        if error[i+1] == error[i]: 
            # add random perturbations. pick n x,y pairs randomly... 
            x = np.random.randint(P.shape[0], size=n)
            y = np.random.randint(P.shape[1], size=n)
            
            for j in range(len(x)):
                P[x[j],y[j]] = np.random.randint(M) # ...and flip to random position in Hc (minicolumn)
            
            
        
        elif error[i+1] < best:
            Pbest = P.copy()    # TODO: remove? the plotting uses the last P as for now...
            best = error[i+1]
        
        
        # randomly shuffle columns
        np.random.shuffle(P.T)
        
    
    # Added while statement so that the updating without shuffeling goes on until 
    #   the error is fixed (not changing)    
    e = error[-1]
    eprev = 100
    error = list(error)
    while not eprev == e: 
        P = greedy_update(Dn, P, H, M)
        eprev = e
        e = hamming_distance(P,Dn,H)
        
        error.append(e)
        
        print(eprev, e)
    
    if saveP:
        
        if e < emax:
            # save patterns...
            # OBS! Pbest have been exchanged for P here!
            if save_fname:
                with open('{}.json'.format(save_fname), 'w') as f:
                    json.dump(P.astype(int).tolist(), f)
            else:
                with open('OutputPatterns/patterns_average.json', 'w') as f:
                    json.dump(P.astype(int).tolist(), f) 
            # and plot
            plot_(Dn,P, H, error=error, saveFig=1, save_fname=save_fname)
            
        else:
            print('--- error={:.2f} > {} --- reinitiating -----'.format(e, emax))
            P = set_patterns(Dn, H, M, Nr=5)
            upd_pattern(P,Dn,H,M,saveP=1,save_fname=save_fname, emax=emax)
            



@jit(nopython=True)
def greedy_update(D, P, H, M):
    # loop and fill the pattern matrix column-vice (one Hc at a time)
    #   **** This is the core of the algorithm ****
    
    npat = len(D)
    
    # loop over patterns one Hc at the time
    for j in range(H):
        for i in range(npat):
            
            # calc optimal position of hypercolumn j in pattern i 
            #       (as the other patterns look right now)
            C = np.zeros(M)
            for m in range(M):
                
                P[i,j] = m
                
                # create matrix with hamming distances of all patterns in P
                nc = 0
                s  = 0
                for ii in range(npat):
                    for jj in range(ii):
                        o = 0
                        for h in range(H):
                            if P[ii,h] == P[jj,h]: o += 1
                        
                        # calc distance to D:
                        #   1. number of Hc's minus the overlap -> hamming distance. 
                        #   2. hamming distance minus target distance (from D) in absolute terms
                        h   = (H - o) - D[ii,jj]
                        if h < 0: h = h*(-1)
                        s  += h 
                        nc += 1
                
                # normalize and save
                C[m] = s / nc
                
            
            # select the value with the shortest distance working with numba?
            P[i,j] = np.argsort(C)[0]
    
    return P


def set_patterns(D, H, M, P=[], Nr=0):
    # fill pattern matrix column-vice (one Hc at a time)
    #   **** This is the core of the algorithm ****
    # -> adds a portion af randomly set columns in each hypercolumn
    
    npat = len(D)
    
    # initialize
    if not len(P):
        P       = np.random.uniform(M, size=(npat,H))
        P[0]    = np.zeros(H)
    
    P = greedy_update(D, P, H, M)
    
    # randomly perturb patterns?
    if Nr:
        
        # create random matrix with same shape
        R = np.random.randint(M, size=P.shape)
        
        # create matrix with zeros and ones used to select subportions of P and R
        r01 = np.ones(P.shape)
        r01[:Nr,:] = 0
        r01 = np.random.permutation(r01.flatten()).reshape(P.shape)
        
        P = P*r01 + R*(1-r01)
    
    return P



def calc_energy(x):
    
    return -1/2 * np.sum(x)




def hamming_distance(a,D, H, return_distances=0):
    
    '''
    a holds the index of the minicolumns for each hypercolumn and pattern.
        if two patterns hold the same value for a single Hc, they have an overlap of 1.
        -e.g. [1,2,3] and [2,2,3] would have an overlap of 2
        
        The distance is then the number of hypercolumns (H) - overlap, e.g. 3 - 2 = 1   
    '''
    
    # TODO perhaps there is some more efficient algorithm for this?
    #   e.g. this one:
    #
    #       d = (a[:, None, :] != a).sum(2) #/ a.shape[1]
    #
    #   The above formula has been tested and gives the same result. UPDATE?
    
    npat    = a.shape[0]
    d       = np.zeros((npat,npat))
    
    for i in range(npat):
        for j in range(i+1):
            
            d[i,j] = H - np.sum([1 for jj in range(a.shape[1]) if a[i,jj] == a[j,jj]])
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
    

def plot_(D,P, H, error=[], saveFig=0, save_fname='patfig'):
    
    d,e = hamming_distance(P,D, H, return_distances=1 )
    print()
    print(e)
    
    fig,ax = plt.subplots(2,3, figsize=(10,8))
    
    ax[0,0].imshow(D, interpolation='none', aspect='equal', vmin=0, vmax=H, cmap='Greys_r')
    ax[0,0].set_title('Original distances')
    plot_patterns(P, ax=ax[0,1], H=H)
    ax[0,1].set_title('Resulting patterns')
    #ax[0,1].set_ylim([-1,10])
    ax[0,2].imshow(d, interpolation='none',aspect='equal', vmin=0, vmax=H, cmap='Greys_r')
    ax[0,2].set_title('From patterns. HE = {}'.format(np.around(e,2)))
    
    ax[1,2].imshow(np.abs(d-D), interpolation='none',aspect='equal', vmin=0, vmax=H, cmap='Greys')
    ax[1,2].set_ylabel('diff |Org-dPatterns|')
    
    ax[1,1].axis('off')
    
    if len(error):
        ax[1,0].plot(error)
        ax[1,0].set_ylabel('Hamming error')
        #ax[1,0].set_ylim([0,1])
    '''
    for i in [0,2]: 
        ax[0,i].set_yticks([0,5,10,15])
        ax[0,i].set_ylim([15.5,-0.5])
    ax[1,2].set_yticks([0,5,10,15])
    ax[1,2].set_ylim([15.5,-0.5])'''
    
    fig.suptitle(save_fname.split('/')[-1])
    
    
    if saveFig:
        if save_fname:
            plt.savefig(save_fname, dpi=300)
        else:
            plt.savefig('OutputPatterns/patterns_average', dpi=300)


def plot_patterns(h, ax=None, H=None):
    
    N = len(h)
    Cgrad = [[0,n/(N-1),n/(N-1)] for n in range(N)]
    
    if not ax:
        fig,ax = plt.subplots(1,1)
    for i in range(h.shape[0]):
        ax.plot(range(H), h[i], '.-', 
                        ms=16, mec='k', lw=1, markerfacecolor='w',
                        label='p{}'.format(i), c=Cgrad[i])
    
    ax.set_xlabel('hypercolumns')
    ax.set_ylabel('microcolumns')



def sort_and_compress():
    '''
    sort vectors from high to low overlap.
        i.e. 
        make sure the first positions have the largest spread and the last the smallest:
            spread  pos 1: [0,1,3,4], 
                    pos 2: [0,2,3], 
                    ... , 
                    pos N: [0,1] 
    
    also compress so that the lowest "range" possible is used in the vectors 
        i.e. 
        if all the vectors have one of the following numbers in the first position 0,1,3,4:
            shift 3 -> 2 and 4 -> 3 to obtain 0,1,2,3
    '''
    #TODO
    pass   



# if run from terminal...   ===============================================================
if __name__ == "__main__":
    
    if len(sys.argv) > 1: 
        if sys.argv[1].isnumeric():
            print('inne', sys.argv[1])
            main(emax=float(sys.argv[1]))
        else:
            print('ERROR: accept single numeric argument only.\nexample run:\n\tpython gmc_sorting.py\nor\n\tpython gmc_sorting.py 0.4')
            print('\ndefault maximal error (emax) = 0.5')
    else:
        main()
    
    
