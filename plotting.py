
import numpy as np
import matplotlib.pyplot as plt
import gmc_sorting as gmc



def plot_(D,P, H, error=[], saveFig=0, save_fname='patfig', vmin=0, vmax=10):
    
    d,e = gmc.hamming_distance(P,D, H, return_distances=1 )
    print()
    print(e)
    
    fig,ax = plt.subplots(2,3, figsize=(10,8))
    
    ax[0,0].imshow(D, interpolation='none', aspect='equal', vmin=vmin, vmax=vmax, cmap='Greys_r')
    ax[0,0].set_title('Original distances')
    plot_patterns(P, ax=ax[0,1], H=H)
    ax[0,1].set_title('Resulting patterns')
    ax[0,2].imshow(d, interpolation='none',aspect='equal', vmin=vmin, vmax=vmax, cmap='Greys_r')
    ax[0,2].set_title('From patterns. HE = {}'.format(np.around(e,2)))
    
    ax[1,2].imshow(np.abs(d-D), interpolation='none',aspect='equal', vmin=vmin, vmax=vmax, cmap='Greys')
    ax[1,2].set_ylabel('diff |Org-dPatterns|')
    
    ax[1,1].axis('off')
    
    if len(error):
        ax[1,0].plot(error)
        ax[1,0].set_ylabel('Hamming error')
    
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

