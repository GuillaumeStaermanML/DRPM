# DRPM: Depth-trimmed Regions based Pseudo-Metric between probability distributions.
=========================================


This repository hosts Python code of the Depth-trimmed Regions based Pseudo-Metric (DRPM) introduced in https://arxiv.org/pdf/2103.12711.

Algorithm
---------

DRPM is a pseudo-metric between probability distributions based on depth-trimmed regions of each distribution. It provides a score in $R_+$ reflecting how close the two distributions are (the smaller it is, the closer it is). 

Some parameters may be eventually set by the user: 

                                - n_alpha: int, default=10
                                    The Monte-Carlo parameter for the approximation of the integral
                                    over alpha.

                                - n_dirs: int, default=100 * n_features
                                    The number of directions for approximating the supremum over
                                    the unit sphere.

                                - data_depth: str in {'tukey', 'projection', 'irw', 'ai_irw'}, default='irw'

                                - eps_min: float in [0,eps_max], default=0
                                    the lowest level set

                                - eps_max: float in [eps_min,1], default=1
                                    the highest level set.



Quick Start :
------------

Create toy training and testing datasets:

.. code:: python

  >>> import numpy as np
  
  >>> n_samples = 1000
  >>> n_features = 10

  >>> X = np.random.randn(n_samples, n_features)
  >>> Y = np.random.randn(n_samples, n_features) + 10

  >>> eps = 0.1            #Epsilon in the paper, it tunes the robustness of the metric.
  >>> n_alpha = 10         #Number of level sets. 
  >>> n_proj = 1000        #Number of projections to approximate depth functions.
  >>> chosen_depth = 'irw' #The chosen depth function in {'tukey', 'irw', 'ai_irw', 'projection'}.
  
  >>> distance = DRPM(X, Y, eps_min=eps, n_alpha=n_alpha, n_dirs=n_proj, data_depth=chosen_depth) # use the algorithm
  >>> print(distance)
                                                               
Dependencies
------------

These are the dependencies to use FIF:

* numpy 
* sklearn

Cite
----

If you use this code in your project, please cite::

@article{staerman2021pseudo,
  title={A Pseudo-Metric between Probability Distributions based on Depth-Trimmed Regions},
  author={Staerman, Guillaume and Mozharovskyi, Pavlo and Cl{\'e}men{\c{c}}on, St{\'e}phan and d'Alch{\'e}-Buc, Florence},
  journal={arXiv preprint arXiv:2103.12711},
  year={2021}
}
