Empirical Likelihood One-Way ANOVA.
==========================

Written on python with the use of ``numpy`` and ``scipy``. Based on the idea of *Empirical Likelihood* developed by A.B. Owen.

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
    
``fit()`` method will produce some output if ``verbose=True``:

.. code:: python

    Residual on call 50:      5.5550491452876355e-11
    Opt. Lagrange mult.:      [ 14.219972 -45.204419  15.579331   4.386927 -10.824406]
    MELE:                     0.24332965083488042
    Probs is positive:        True
    Sum of probs:             1.000000000000325
    The solution converged.
    
A brief explanation of this output: the output includes residual (there is a non-linear equation that we solve numerically using ``scipy``), MELE - **Maximum Empirical Likelihood Estimator** of the common mean. Actually, the test builds a distribution on our samples: so there is probability attached to every value of the sample. We can check if the distribution that we've got is really a distribution (attached probabilities is positive and sums to 1).

After ``fit()`` method you can print some things...

.. code:: python

    print(anova.logR)   # the statistic of test: the log-profile function R(X).
    print(anova.pvalue) # the pvalue of the test.
    print(anova.l)      # optimal Lagrange Multipliers.
    print(anova.MELE)   # Maximum (Empirical) Likelihood Estimator of the common mean.
 
... and find the confidence interval for the common mean (if the test hasn't rejected H_0 hypothesis).

.. code:: python
    
    anova.confidence_interval(X)
