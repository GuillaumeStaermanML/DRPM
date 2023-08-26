# Authors: Guillaume Staerman <staerman.guillaume@gmail.com>
#
# License: MIT

import numpy as np

from utils import standardize, sampled_sphere

########################################################
#################### Data Depths ########################
########################################################  

def tukey_depth(X, n_dirs=None):
    """ Compute the score of the classical tukey depth of X w.r.t. X

    Parameters
    ----------
    X : Array of shape (n_samples, n_features)
            The training set.

    ndirs : int | None
        The number of random directions to compute the score.
        If None, the number of directions is chosen as 
        n_features * 100.
        
    Return
    -------
    tukey_score: Array of float
        Depth score of each delement in X.
    """

    if n_dirs is None:
        n_dirs = n_features * 100

    n_samples, n_features = X.shape

    #Simulated random directions on the unit sphere.  
    U = sampled_sphere(n_dirs, n_features)

    sequence = np.arange(1, n_samples + 1)
    depth = np.zeros((n_samples, n_dirs))

    #Compute projections       
    proj = np.matmul(X, U.T)

    rank_matrix = np.matrix.argsort(proj, axis =0) 

    for k in range(n_dirs):
        depth[rank_matrix[:, k], k] = sequence  

    depth =  depth / (n_samples * 1.)  

    depth_score = np.minimum(depth, 1 - depth)
    tukey_score = np.amin(depth, axis=1)

    return tukey_score


def projection_depth(X, n_dirs=None):
    """ Compute the score of the projection depth of X w.r.t. X

    Parameters
    ----------
    X : Array of shape (n_samples, n_features)
        The training set.

    ndirs : int | None
        The number of random directions to compute the score.
        If None, the number of directions is chosen as 
        n_features * 100.
        
    Return
    -------
    projection score: Array of float
        Depth score of each delement of X.
    """
    
    n, d = X.shape

    if n_dirs is None:
        n_dirs = n_features * 100

    n_samples, n_features = X.shape

    #Simulated random directions on the unit sphere.  
    U = sampled_sphere(n_dirs, n_features)
 
    #Compute projections
    proj = np.matmul(X, U.T)


    depth = np.zeros((n_samples, n_dirs))
    MAD = np.zeros(n_dirs)

    #Compute stahel-Donoho outlyingness on projections
    med_proj = np.median(proj, axis=0)
    MAD = np.median(np.absolute(proj - med_proj.reshape(1, -1)), axis=0) 
    depth = np.absolute(proj - med_proj.reshape(1, -1) ) / MAD
    outlyingness = np.amax(depth, axis=1)

    projection_score = 1 / (1 + outlyingness) 

    return projection_score


def ai_irw_depth(X,  AI=True, robust=False, n_dirs=None, random_state=None):
    """ Compute the score of the (Affine-Invariant-) Integrated Rank 
        Weighted (AI-IRW) depth of X_test w.r.t. X

    Parameters
    ----------

    X : Array of shape (n_samples, n_features)
            The training set.

    AI: bool
        if True, the affine-invariant version of irw is computed. 
        If False, the original irw is computed.

    robust: bool, default=False
        if robust is true, the MCD estimator of the covariance matrix
        is performed.

    n_dirs : int | None
        The number of random directions needed to approximate 
        the integral over the unit sphere.
        If None, n_dirs is set as 100* n_features.

    random_state : int | None
        The random state.
        
    Return
    -------
    ai_irw_score: Array
        Depth score of each element in X.
    """
    
    if random_state is None:
        random_state = 0

    np.random.seed(random_state)

    if AI:
        X_reduced = standardize(X, robust)
    else:
        X_reduced = X.copy()

    n_samples, n_features = X_reduced.shape 

    if n_dirs is None:
        n_dirs = n_features * 100

    #Simulated random directions on the unit sphere. 
    U = sampled_sphere(n_dirs, n_features)

    sequence = np.arange(1, n_samples + 1)
    depth = np.zeros((n_samples, n_dirs))
    
    proj = np.matmul(X_reduced, U.T)
    rank_matrix = np.matrix.argsort(proj, axis=0)
    
    for k in range(n_dirs):
        depth[rank_matrix[:, k], k] = sequence  
    
    depth =  depth / (n_samples * 1.)         
    depth_score = np.minimum(depth, 1 - depth)
    ai_irw_score = np.mean(depth_score, axis=1)

    return ai_irw_score
    

            
