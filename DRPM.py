# Authors: Guillaume Staerman <staerman.guillaume@gmail.com>
#
# License: MIT

import numpy as np

from utils import  sampled_sphere
from depths import tukey_depth, projection_depth, ai_irw_depth

def DRPM(X, Y, n_alpha=10, data_depth='irw', eps_min=0,
         eps_max=1, p=2, n_dirs=None, random_state=None):
    """The Deph-trimmed Regions based Pseudo-Metric (DRPM) 
       between two probability distributions.

    Parameters
    ----------

    X: array of shape (n_samples, n_features)
        The first sample.

    Y: array of shape (n_samples, n_features)
        The second sample.

    n_alpha: int, default=10
        The Monte-Carlo parameter for the approximation of the integral
        over alpha.

    n_dirs: int
        The number of directions for approximating the supremum over
        the unit sphere.

    data_depth: str in {'tukey', 'projection', 'irw', 'ai_irw'}, default='irw'

    eps_min: float in [0,eps_max]
        the lowest level set.

    eps_max: float in [eps_min,1]
        the highest level set.

    p: int
        the power of the ground cost.

    random_state : int | None
        The random state.

    Return
    ------

    dr_score: float
        the computed pseudo-metric score between X and Y.

    References
    ----------
    G. Staerman, P. Mozharovskyi, P. Colombo, S. Clémençon, Florence d'Alché-Buc. 
    A Pseudo-Metric between Probability Distributions based on 
    Depth-Trimmed Regions. (preprint) 2022. 
    https://arxiv.org/pdf/2103.12711
    
    """

    if random_state is None:
        random_state = 0

    np.random.seed(random_state)

    if data_depth not in {'tukey', 'projection', 'irw', 'ai_irw'}:
        raise NotImplementedError('This data depth is not implemented')

    if eps_min > eps_max:
        raise ValueError('eps_min must be lower than eps_max')

    if eps_min < 0 or eps_min > 1:
        raise ValueError('eps_min must be in [0,eps_max]')

    if eps_max < 0 or eps_max > 1:
        raise ValueError('eps_min must be in [eps_min,1]')

    _, n_features = X.shape

    if n_dirs is None:
        n_dirs = n_features * 100

    if data_depth == "tukey":
        depth_X = tukey_depth(X, n_dirs=n_dirs)
        depth_Y = tukey_depth(Y, n_dirs=n_dirs)
    elif data_depth == "projection":
        depth_X = projection_depth(X, n_dirs=n_dirs)
        depth_Y = projection_depth(Y, n_dirs=n_dirs)
    elif data_depth == "irw":
        depth_X = ai_irw_depth(X, AI=False, n_dirs=n_dirs)
        depth_Y = ai_irw_depth(Y, AI=False, n_dirs=n_dirs)
    elif data_depth == "ai_irw":
        depth_X = ai_irw_depth(X, AI=True, robust=True, n_dirs=n_dirs)
        depth_Y = ai_irw_depth(Y, AI=True, robust=True, n_dirs=n_dirs)


    # draw 'n_dirs' vectors of the unit sphere in dimension 'n_features'.
    U = sampled_sphere(n_dirs, n_features)
    proj_X = np.matmul(X, U.T)
    proj_Y = np.matmul(Y, U.T)

    liste_alpha = np.linspace(int(eps_min * 100), int(eps_max * 100), n_alpha)
    quantiles_DX = [np.percentile(depth_X, j) for j in liste_alpha]
    quantiles_DY = [np.percentile(depth_Y, j) for j in liste_alpha]

    dr_score = 0
    for i in range(n_alpha):
        d_alpha_X = np.where(depth_X >= quantiles_DX[i])[0]
        d_alpha_Y = np.where(depth_Y >= quantiles_DY[i])[0]
        supp_X = np.max(proj_X[d_alpha_X], axis=0)
        supp_Y = np.max(proj_Y[d_alpha_Y], axis=0)
        dr_score += np.max((supp_X - supp_Y) ** p)

    return (dr_score / n_alpha) ** (1 / p)