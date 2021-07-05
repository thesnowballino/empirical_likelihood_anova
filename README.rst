Empirical Likelihood ANOVA.
==========================

Written on python with the use of ``numpy`` and ``scipy``. Based on the idea of Empirical Likelihood developed by A.B. Owen.

Example.
-------

.. code:: python

    from empirical_likelihood_anova import ANOVA_EL
    import numpy as np
    import scipy.stats
    
    anova = ANOVA_EL()

    n = [50, 20, 20, 10, 15]
    K = 5
    
    X = [scipy.stats.norm.rvs(size=n[i]) for i in range(K)] # a list of numpy arrays with shapes (n_i, ) 
    
    anova.fit(X, verbose=True)

After ``fit()`` method you are able to print some things:

.. code:: python

    print(anova.logR)   # the statistic of test: the log-profile function R(X).
    print(anova.pvalue) # the pvalue of the test.
    print(anova.l)      # optimal Lagrange Multipliers.
    print(anova.MELE)   # Maximum (Empirical) Likelihood Estimator of the common mean.
